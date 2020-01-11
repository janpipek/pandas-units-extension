import builtins
import operator
import re
import sys
from typing import Any, Union

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
from pandas.core.dtypes.generic import ABCIndexClass, ABCSeries

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
        if string == cls.BASE_NAME:
            return cls()
        match = re.match(f"{cls.BASE_NAME}\\[(?P<name>.*)\\]$", string)
        if not match:
            raise TypeError(f"Invalid UnitsDtype string: '{string}'")
        return cls(match["name"])

    @classmethod
    def construct_array_type(cls) -> builtins.type:
        """Associated extension array."""
        return UnitsExtensionArray

    @property
    def name(self) -> str:
        return f"{self.BASE_NAME}[{self.unit.to_string()}]"

    def __repr__(self):
        return f'{self.__class__.__name__}("{self.unit.to_string()}")'


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


def as_quantity(obj: Any) -> Quantity:
    """Try to convert whatever input to a Quantity."""
    if isinstance(obj, Quantity):
        return obj
    elif isinstance(obj, UnitsExtensionArray):
        return Quantity(obj.value, obj.unit)
    elif is_array_like(obj) and obj.dtype == "timedelta64[ns]":
        # Note: Timedelta is internally represented as int64
        return Quantity(np.asarray(obj, dtype=np.int64), "ns").to("s")
    elif is_list_like(obj):
        obj = list(obj)
        if len(obj) == 0:
            return Quantity([], "")
        elif all(isinstance(item, str) for item in obj):
            return Quantity([Quantity(item) for item in obj])
    return Quantity(obj)


class UnitsExtensionArray(ExtensionArray, ExtensionScalarOpsMixin):
    """Pandas extension array supporting physical quantities with units."""

    def __init__(
        self, array, unit: Union[None, str, Unit] = None, *, copy: bool = True
    ):
        if isinstance(array, Quantity):
            if copy:
                array = array.copy()
            self._dtype = UnitsDtype(array.unit)
            self._value = array.value.astype(float)
        else:
            q = as_quantity(array)
            if q.unit.is_unity():
                if unit:
                    q = q * unit
            else:
                if unit and q.unit != unit:
                    raise ValueError("Dtypes are not equivalent")

            self._dtype = UnitsDtype(q.unit)
            self._value = q.value.astype(float)

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

    def __array__(self, dtype=None) -> np.ndarray:
        """Implicit conversion to numpy array."""
        return self.value.astype(dtype) if dtype else self.value

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

        if isinstance(dtype, UnitsDtype):
            return _as_units_dtype(dtype.unit)
        elif dtype == "timedelta64[ns]":
            nanoseconds = convert(as_quantity(self), "ns")
            return np.array(nanoseconds.value, dtype="timedelta64[ns]", copy=copy)
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

    def _formatter(self, boxed: bool = False):
        """Formatter to always include unit name in the output.

        TODO: Not sure if this is the best (differ on boxed?)
        """
        return lambda x: (str(x) if isinstance(x, Quantity) else f"{x} {self.unit}")

    def __getitem__(self, item):
        if np.isscalar(item):
            return Quantity(self.value[item], unit=self.unit)
        else:
            return self.__class__(self.value[item], unit=self.unit)

    def __setitem__(self, key, value):
        if is_scalar(value) and np.isnan(value):
            q = Quantity(value, self.unit)
        elif is_list_like(value) and len(value) == 0:
            return
        else:
            q = as_quantity(value)
        q = convert(q, self.unit)
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
    def _create_method(cls, op, coerce_to_dtype=True):
        # Overriden from the default variant
        # to by-pass conversion to numpy arrays.
        def _invalid_operator():
            if is_equality:
                return NotImplemented
            else:
                raise TypeError

        def _binop(self, other):
            if isinstance(other, (ABCSeries, ABCIndexClass)):
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
            if coerce_to_dtype:
                return cls(result_q)
            else:
                return result_q

        # Get info about the operator
        op_name = ops._get_op_name(op, True)
        is_comparison = op_name in [
            "__eq__",
            "__ne__",
            "__lt__",
            "__gt__",
            "__le__",
            "__ge__",
        ]
        is_equality = op_name in ["__eq__", "__ne__"]

        return set_function_name(_binop, op_name, cls)

    def copy(self, deep=False) -> "UnitsExtensionArray":
        return self.__class__(self.value, self.unit, copy=True)

    def _reduce(self, name, skipna=True, **kwargs):
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
            return getattr(q, name)(**kwargs)

        elif name in to_nanops:
            data = self.value
            method = getattr(nanops, "nan" + name)
            result_without_dim = method(data, skipna=skipna)
            return Quantity(result_without_dim, self.unit)

        elif name in to_error:
            raise TypeError(f"Cannot perform '{name}' with type '{self.dtype}'")

        elif name in to_implement_yet:
            raise NotImplementedError

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
        new_unit = Unit(formatter.to_string(unit.si))
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
