import builtins
import operator
import re
import sys
from typing import Any, Union
import warnings

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.units import Quantity, Unit, imperial
from astropy.units.format.generic import Generic
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    ExtensionScalarOpsMixin,
    register_dataframe_accessor,
    register_extension_dtype,
    register_series_accessor,
)
from pandas.api.types import is_array_like, is_list_like, is_scalar
from pandas.compat import set_function_name
from pandas.core import nanops, ops
from pandas.core.algorithms import take
from pandas.core.dtypes.generic import ABCIndex, ABCSeries, ABCDataFrame
from pandas.core.indexers import (
    check_array_indexer,
    getitem_returns_view,
)
from pandas.util._exceptions import find_stack_level
from pandas._typing import DtypeObj

# Imperial units enabled by default
imperial.enable()


class InvalidUnitConversion(ValueError):
    """The unit cannot be converted to another one."""


class InvalidUnit(ValueError):
    """The unit does not exist."""


@register_extension_dtype
class UnitsDtype(ExtensionDtype):
    """Description of the units type.

    The name is formed as "unit[.*]" where the inside of the square
    brackets must be a unit name as understood by astropy units.
    """

    BASE_NAME = "unit"

    type = Quantity
    kind = "O"

    _is_numeric = False
    _metadata = ("unit",)

    def __init__(self, unit: Union[None, str, Unit] = None):
        if isinstance(unit, (Unit, type(None))):
            self.unit = unit
        else:
            self.unit = Unit(unit)

    @classmethod
    def construct_from_string(cls, string: str) -> "UnitsDtype":
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        if string == cls.BASE_NAME:
            return cls()
        match = re.match(f"{cls.BASE_NAME}\\[(?P<name>.*)\\]$", string)
        if not match:
            raise TypeError(f"Cannot construct a 'UnitsDtype' from '{string}'")
        return cls(match["name"])

    @classmethod
    def construct_array_type(cls) -> builtins.type:
        """Associated extension array."""
        return UnitsExtensionArray

    @property
    def name(self) -> str:
        return f"{self.BASE_NAME}[{self.unit.to_string()}]"

    @property
    def na_value(self) -> Quantity:
        return u.Quantity(np.nan, self.unit)

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.unit.to_string()}")'
    
    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        """
        Return the common dtype, if one exists.

        Used in `find_common_type` implementation. This is for example used
        to determine the resulting dtype in a concat operation.

        If no common dtype exists, return None (which gives the other dtypes
        the chance to determine a common dtype). If all dtypes in the list
        return None, then the common dtype will be "object" dtype (this means
        it is never needed to return "object" dtype from this method itself).

        Parameters
        ----------
        dtypes : list of dtypes
            The dtypes for which to determine a common dtype. This is a list
            of np.dtype or ExtensionDtype instances.

        Returns
        -------
        Common dtype (np.dtype or ExtensionDtype) or None
        """
        if len(set(dtypes)) == 1:
            # only itself
            return self

        # Check that all dtypes are UnitsDtype
        if not all([isinstance(t, UnitsDtype) for t in dtypes]):
            return None

        # Check that the units of all UnitsDtype have the same physical type as self and are therefore convertible to self
        phy_type = self.unit.physical_type
        if all([t.unit.physical_type == phy_type for t in dtypes]):
            return self

        # Different physical types, no common dtype
        return None


def convert(q: Quantity, new_unit: Union[str, Unit], equivalencies=None) -> Quantity:
    """Convert quantity to a new unit.

    :raises InvalidUnit: When target unit does not exist.
    :raises InvalidUnitConversion: If the conversion is invalid.

    Customized to be a bit more universal than the original quantities.
    """
    try:
        return q.to(new_unit, equivalencies or [])
    except u.UnitConversionError:
        if q.unit.physical_type == "temperature":
            return q.to(new_unit, u.temperature())
        else:
            raise InvalidUnitConversion(
                f"Cannot convert unit '{q.unit}' to '{new_unit}'."
            ) from None
    except ValueError as err:
        raise InvalidUnit(f"Unit '{new_unit}' does not exist.") from None


def as_quantity(obj: Any, copy: bool = True) -> Quantity:
    """Try to convert whatever input to a Quantity."""
    if isinstance(obj, Quantity):
        return Quantity(obj, copy=copy)
    elif isinstance(obj, UnitsExtensionArray):
        return Quantity(obj.value, obj.unit, copy=copy)
    elif is_array_like(obj) and obj.dtype == "timedelta64[ns]":
        # Note: Timedelta is internally represented as int64
        return Quantity(np.asarray(obj, dtype=np.int64), "ns", copy=copy).to("s")
    elif is_list_like(obj):
        obj = list(obj)
        copy = False  # Already copied by list()
        if len(obj) == 0:
            return Quantity([], "")
        elif all(isinstance(item, str) for item in obj):
            return Quantity([Quantity(item) for item in obj])
    if copy and hasattr(obj, "copy"):
        obj = obj.copy()
    return Quantity(obj)


class UnitsExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
    """Pandas extension array supporting physical quantities with units."""

    # Adapted from MaskedArray to create new objects of UnitsExtensionArray for views and slices
    @classmethod
    def _simple_new(cls, values: np.ndarray, dtype: UnitsDtype) -> "UnitsExtensionArray":
        """Create a new UnitsExtensionArray from the given values and dtype.

        Parameters
        ----------
        values : np.ndarray
            The numerical values (without unit).
        dtype : UnitsDtype
            The dtype of the new array, which contains the unit information.

        Returns
        -------
        UnitsExtensionArray
            A new UnitsExtensionArray with the given values and dtype.
        """
        result = UnitsExtensionArray.__new__(cls)
        result._dtype = dtype
        result._value = values
        return result

    def __init__(
        self, array, unit: Union[None, str, Unit] = None, *, copy: bool = True
    ):
        q: Quantity = as_quantity(array, copy=copy)

        # Special handling for boolean arrays. Quantity does support boolean arrays, 
        # treating `True` as `1` and `False` as `0`, but this is not sensible here 
        # and can lead to rather unexpected behavior: `u.Quantity(True, u.m) == 1 * u.m` -> `True`.
        # The only expected path here is from pandas.Series.combine where boolean arrays appear 
        # after comparison operations. The raised ValueError is caught there and handled properly.
        if isinstance(q.dtype, np.dtypes.BoolDType):
            raise ValueError("Boolean array cannot sensible be converted to Quantity and therefore UnitsExtensionArray.")

        if isinstance(unit, str):
            unit: Unit = Unit(unit)

        if q.unit.is_unity():
            if unit:
                q: Quantity = q * unit
        elif unit and q.unit != unit:
            # Convert to target unit given by dtype as long as physical types match
            try:
                q: Quantity = convert(q, unit)
            except InvalidUnitConversion as e:
                raise InvalidUnitConversion("Could not convert units in initialization of UnitsExtensionArray: ") from e

        self._dtype: UnitsDtype = UnitsDtype(q.unit)
        self._value: np.ndarray[np.float64] = q.value.astype(float)

    @property
    def value(self) -> np.ndarray:
        """The numerical values (without unit)."""
        return self._value

    @property
    def unit(self) -> Unit:
        """The unit itself."""
        return self.dtype.unit

    @property
    def dtype(self) -> UnitsDtype:
        return self._dtype

    def __len__(self) -> int:
        return len(self.value)

    def __array__(self, dtype=object, copy=None) -> np.ndarray:
        """Implicit conversion to numpy array."""
        # Create array depending on dtype
        if dtype == object:
            if copy == False:
                raise ValueError("Cannot return object array without copy, as each element has to be its own Quantity object.")
            arr = np.array(list(as_quantity(self)), dtype=object)
            # Converting self first to a Quantity and then to a ndarray requires a copy, so copy flag will be set to True
            copy = True
        elif dtype:
            arr = self.value.astype(dtype, copy=copy)
        else:
            arr = np.asarray(self.value, copy=copy)

        # Set writable flag depending on self._readonly and only when no copy was made
        if self._readonly and copy is not True:
            arr.setflags(write=False)

        return arr

    @property
    def nbytes(self) -> int:
        return self.value.nbytes + sys.getsizeof(self.unit)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False) -> "UnitsExtensionArray":
        if dtype:
            result = cls(scalars, unit=dtype.unit, copy=copy)
        else:
            result = cls(scalars, copy=copy)
        return result

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, dtype=None, copy=False
    ) -> "UnitsExtensionArray":
        values = [Quantity(s) for s in strings]
        unit = dtype.unit if dtype else None
        return UnitsExtensionArray(values, unit)

    @classmethod
    def _from_scalars(cls, scalars, *, dtype=None) -> "UnitsExtensionArray":
        """
        Contrary to the superclass, this function will ignore the `dtype`
        if given, and will always try to infer the `dtype` from the scalars.
        This is due to arithmetic operation that are changing the unit and
        thereby the `dtype`. The `dtype` giving to this function comes from
        `self.dtype` in `_cast_pointwise_result`, but `self.dtype` is only the
        original `dtype` of the left operand, not necessarily the correct
        `dtype` of the result.

        Parameters
        ----------
        scalars : sequence
        dtype : ExtensionDtype

        Raises
        ------
        TypeError or ValueError

        Notes
        -----
        This is called in a try/except block when casting the result of a
        pointwise operation in `ExtensionArray._cast_pointwise_result`.
        """
        try:
            result: UnitsExtensionArray = cls._from_sequence(scalars)
        except (ValueError, TypeError):
            raise
        except Exception:
            warnings.warn(
                "_from_scalars should only raise ValueError or TypeError. "
                "Consider overriding _from_scalars where appropriate.",
                stacklevel=find_stack_level(),
            )
            raise

        return result

    def to_quantity(self) -> Quantity:
        """Convert to native Quantity."""
        return as_quantity(self)

    def unique(self) -> "UnitsExtensionArray":
        """Unique values."""
        return self.__class__(pd.unique(self.value), unit=self.unit)

    def to(
        self, new_unit: Union[str, Unit], equivalencies=None
    ) -> "UnitsExtensionArray":
        """Convert to another unit (if possible)."""
        q = self.to_quantity()
        new_data = convert(q, new_unit, equivalencies)
        return UnitsExtensionArray(new_data)

    def astype(self, dtype, copy: bool = True):
        """Convert to a different dtype."""

        def _as_units_dtype(unit):
            return self.to(unit)

        if dtype == self.dtype:
            return self.copy() if copy else self
        elif isinstance(dtype, UnitsDtype):
            return _as_units_dtype(dtype.unit)
        elif dtype == "timedelta64[ns]":
            nanoseconds: Quantity = convert(as_quantity(self, copy=copy), "ns")
            return np.asarray(nanoseconds.value, dtype="timedelta64[ns]")
        elif isinstance(dtype, (str, pd.StringDtype)):
            str_values = [str(q) for q in self.to_quantity()]
            return pd.Series(str_values, dtype=dtype).array
        elif dtype in ["O", "object", object]:
            return np.array([x * self.unit for x in self.value], dtype=object)
        elif isinstance(dtype, str):
            try:
                dtype = UnitsDtype(dtype)
                return _as_units_dtype(dtype.unit)
            except Exception:
                pass

        # Fall-back to default variant
        return ExtensionArray.astype(self, dtype, copy=copy)

    def view(self, dtype=None) -> "UnitsExtensionArray":
        """Create a new object with same data behind it."""
        # TODO: Useful also for 0.25???
        if dtype is not None:
            # TODO: Perhaps implement?
            raise NotImplementedError(dtype)
        result = UnitsExtensionArray.__new__(UnitsExtensionArray)
        result._dtype = self.dtype
        result._value = self.value
        result._readonly = self._readonly
        return result

    def _formatter(self, boxed: bool = False):
        """Formatter to always include unit name in the output.

        TODO: Not sure if this is the best (differ on boxed?)
        """
        return lambda x: (str(x) if isinstance(x, Quantity) else f"{x} {self.unit}")

    def __getitem__(self, item):
        # Return zerodim Quantity object for singular item
        if is_scalar(item):
            return Quantity(self.value[item], unit=self.unit)

        # Use pandas utility function to check and convert the item to a valid indexer
        item = check_array_indexer(self, item)

        # Create new UnitsExtensionArray
        result: UnitsExtensionArray = self._simple_new(self.value[item], self.dtype)

        # If the result is a view, keep read-only flag
        if getitem_returns_view(self, item):
            result._readonly = self._readonly

        return result

    def __setitem__(self, key, value):
        # Return early if value is empty list or None, as this is a no-op for __setitem__
        if (is_list_like(value) and len(value) == 0) or value is None:
            return

        # If readonly flag is set array cannot be modified in place, dispatch to pandas to comply with CoW
        if self._readonly:
            raise ValueError("Cannot modify read-only array")

        # Convert NaN to Quantity with correct unit
        if is_scalar(value) and np.isnan(value):
            value = Quantity(value, self.unit)

        # Use pandas utility function to check and convert the item to a valid indexer
        key = check_array_indexer(self, key)

        # Convert value to quantity and convert to same unit as self if necessary
        q: Quantity = as_quantity(value)
        q: Quantity = convert(q, self.unit)

        # Set the values at the given key to the numerical values of the quantity
        self.value[key] = q.value

    def take(self, indices, allow_fill=False, fill_value=None) -> "UnitsExtensionArray":
        """Integer-based selection of items."""
        if allow_fill:
            if fill_value is None or np.isnan(fill_value):
                fill_value = np.nan
            else:
                fill_value = fill_value.value
        values = take(self.value, indices, allow_fill=allow_fill, fill_value=fill_value)
        return UnitsExtensionArray(values, self.unit)

    @classmethod
    def _concat_same_type(cls, to_concat) -> "UnitsExtensionArray":
        # TODO: This does not work for similar types.
        if len(to_concat) == 0:
            return cls([])
        elif len(to_concat) == 1:
            return to_concat[0]
        elif len(set(item.unit for item in to_concat)) != 1:
            # TODO: And this actually never happens.
            raise ValueError("Not all concatenated arrays have the same units.")
        else:
            return cls(
                np.concatenate([item.value for item in to_concat]), to_concat[0].unit
            )

    def isna(self):
        return np.isnan(self.value)

    @classmethod
    def _create_method(cls, op, coerce_to_dtype=True, result_dtype=None):
        # Overriden from the default variant
        # to by-pass conversion to numpy arrays.

        # Get info about the operator
        op_name = getattr(op, "__name__", str(op))
        is_comparison = op_name in [
            "eq", "__eq__",
            "ne", "__ne__",
            "lt", "__lt__",
            "gt", "__gt__",
            "le", "__le__",
            "ge", "__ge__",
        ]
        is_equality = op_name in ["eq", "ne", "__eq__", "__ne__"]
        is_divmod = op_name in ["divmod", "__divmod__", "rdivmod", "__rdivmod__"]
        
        def _invalid_operator():
            if is_equality:
                return NotImplemented
            else:
                raise TypeError

        def _binop(self, other):
            if isinstance(other, (ABCIndex, ABCSeries, ABCDataFrame)):
                # rely on pandas to unbox and dispatch to us
                return NotImplemented

            elif is_scalar(other):
                if is_comparison:
                    return NotImplemented

            elif is_array_like(other):
                if not isinstance(other.dtype, UnitsDtype):
                    if is_comparison:
                        return _invalid_operator()

            # Convert the thing to quantities
            self_q = as_quantity(self)
            other_q = as_quantity(other)

            if is_comparison:
                # Try apply conversion (we need same type for comparisons)
                if is_array_like(other) and other.dtype != self.dtype:
                    try:
                        other_q = convert(other_q, self.unit)
                    except InvalidUnitConversion:
                        return _invalid_operator()

            result_q = op(self_q, other_q)

            # Divmod returns tuple of two Quantity objects and they have to be handled separately
            if is_divmod:
                if coerce_to_dtype:
                    return cls(result_q[0]), cls(result_q[1])
                return result_q[0], result_q[1]
            else:
                if coerce_to_dtype:
                    return cls(result_q)
                return result_q

        return set_function_name(_binop, op_name, cls)

    def copy(self, deep=False) -> "UnitsExtensionArray":
        return self.__class__(self.value, self.unit, copy=True)

    def _reduce(self, name, skipna=True, keepdims=False, **kwargs):
        """Implementation of pandas basic reduce methods."""
        # Borrowed from IntegerArray

        to_proxy = ["min", "max", "sum", "mean", "std", "var"]
        to_nanops = ["median", "sem"]
        to_error = ["any", "all", "prod"]

        # TODO: Check the dimension of this
        to_implement_yet = ["kurt", "skew"]

        if name in to_proxy:
            q = self.to_quantity()
            if name in ["std", "var"]:
                kwargs = {"ddof": kwargs.pop("ddof", 1)}
            else:
                kwargs = {}
            if skipna:
                q = q[~np.isnan(q)]
            result = getattr(q, name)(**kwargs)

        elif name in to_nanops:
            data = self.value
            method = getattr(nanops, "nan" + name)
            result_without_dim = method(data, skipna=skipna)
            result = Quantity(result_without_dim, self.unit)

        elif name in to_error:
            raise TypeError(f"Cannot perform '{name}' with type '{self.dtype}'")

        elif name in to_implement_yet:
            raise NotImplementedError

        if keepdims:
            return self._from_scalars([result], dtype=self.dtype)
        return result

    def _values_for_factorize(self) -> tuple[np.ndarray, Any]:
        """Generate values for factorization"""
        return self.value, None

    @classmethod
    def _from_factorized(cls, values, original) -> "UnitsExtensionArray":
        return UnitsExtensionArray(values, original.dtype.unit)

    def value_counts(
        self, normalize=False, sort=True, ascending=False, bins=None, dropna=True
    ) -> pd.Series:
        # TODO: Is it possible to include units? We'd need custom index
        return pd.Index(self.value).value_counts(
            normalize, sort, ascending, bins, dropna
        )


@register_series_accessor("units")
class UnitsSeriesAccessor:
    """Accessor adding unit functionality to series."""

    def __init__(self, obj):
        # Inspired by fletcher
        if not isinstance(obj.array, UnitsExtensionArray):
            raise AttributeError("Only UnitsExtensionArray has units accessor.")
        self.obj = obj

    @property
    def unit(self) -> Unit:
        """The Series' unit."""
        return self.obj.array.unit

    def _wrap(self, result: UnitsExtensionArray) -> pd.Series:
        """Construct a series with different data but same index and name."""
        return self.obj.__class__(result, name=self.obj.name, index=self.obj.index)

    def to(self, unit, equivalencies=None) -> pd.Series:
        """Convert series to another unit."""
        new_array = self.obj.array.to(unit, equivalencies)
        return self._wrap(new_array)

    def to_si(self) -> pd.Series:
        """Convert series to a relevant SI unit."""
        unit = self.obj.array.unit
        formatter = Generic()
        formatter._show_scale = False
        new_unit = Unit(formatter.to_string(unit.si.bases[0]))
        return self.to(new_unit)


@register_dataframe_accessor("units")
class UnitsDataFrameAccessor:
    """Accessor adding unit functionality to data frames."""

    def __init__(self, obj: pd.DataFrame):
        self.obj = obj

    def to_si(self) -> pd.DataFrame:
        """Convert all columns that are of unit type to SI."""

        def _f(col):
            try:
                return col.units.to_si()
            except AttributeError:
                return col

        return self.obj.apply(_f)


UnitsExtensionArray._add_arithmetic_ops()
UnitsExtensionArray._add_comparison_ops()

UnitsExtensionArray.__pow__ = UnitsExtensionArray._create_method(operator.pow)
