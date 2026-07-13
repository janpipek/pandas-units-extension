"""
The ``astropy`` umbrella: base dtype, base array, accessors and ``from_astropy``.

This module owns the generic machinery shared by every astropy-backed extension
type. Concrete types (e.g. :mod:`~pandas_units_extension.quantity`) subclass
:class:`AstropyDtype`/:class:`AstropyExtensionArray`/:class:`AstropySeriesAccessor`
and register themselves through :mod:`~pandas_units_extension.registry`.

Only the umbrella :class:`AstropyDtype` is registered with pandas. Its
``construct_from_string`` is the single parser for the ``astropy{...}[...]``
grammar and dispatches through the registry to the concrete dtype classes.
"""

from typing import Any

import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_dataframe_accessor,
    register_extension_dtype,
    register_series_accessor,
)

from ._grammar import parse_dtype_string
from .registry import (
    get_by_selector,
    infer_from_array,
    infer_from_object,
    infer_from_params,
)


@register_extension_dtype
class AstropyDtype(ExtensionDtype):
    """
    Umbrella dtype for astropy-backed extension arrays.

    The bare ``astropy`` string resolves to an (unresolved) instance of this
    class, whose array type infers the concrete type from the data. Typed
    strings like ``astropy{quantity}[m]`` are parsed here and dispatched through
    the registry to a concrete :class:`AstropyDtype` subclass. A selector-less
    ``astropy[m]`` resolves the concrete type from the parameters instead.

    Examples
    --------
    >>> import pandas as pd
    >>> import astropy.units as u
    >>> import pandas_units_extension  # registers the dtype with pandas
    >>> pd.Series([1, 2, 3], dtype="astropy{quantity}[m]").dtype
    QuantityDtype("m")
    >>> pd.Series([1 * u.m, 2 * u.m], dtype="astropy").dtype  # inferred from data
    QuantityDtype("m")
    """

    type: type = object
    kind: str = "O"
    _is_numeric: bool = False

    @property
    def name(self) -> str:
        return "astropy"

    @classmethod
    def construct_from_string(cls, string: str) -> "AstropyDtype":
        """
        Parse an ``astropy`` dtype string and dispatch to the concrete dtype.

        Parameters
        ----------
        string : str
            The dtype string, e.g. ``"astropy"``, ``"astropy{quantity}"``,
            ``"astropy{quantity}[m]"`` or ``"astropy[m]"`` (type inferred from
            the parameters).

        Returns
        -------
        AstropyDtype
            The unresolved umbrella dtype for the bare ``"astropy"`` string, or a
            concrete subclass instance (e.g. ``QuantityDtype``) for a typed one
            or a parameter-only (``astropy[m]``) one.

        Raises
        ------
        TypeError
            If the string is not a valid ``astropy`` dtype string, names an
            unknown type selector, or has parameters that no registered type can
            parse.

        Examples
        --------
        >>> AstropyDtype.construct_from_string("astropy{quantity}[m]")
        QuantityDtype("m")
        >>> AstropyDtype.construct_from_string("astropy[m]")  # type inferred
        QuantityDtype("m")
        >>> type(AstropyDtype.construct_from_string("astropy")).__name__
        'AstropyDtype'
        """
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        try:
            parsed = parse_dtype_string(string)
        except TypeError:
            # Not an astropy dtype string (or malformed): let pandas' registry
            # fall through to other dtypes.
            raise TypeError(
                f"Cannot construct a '{cls.__name__}' from '{string}'"
            ) from None

        if parsed.selector is None:
            if parsed.params is None:
                # Bare "astropy": unresolved umbrella, type inferred from data.
                return AstropyDtype()
            # "astropy[...]": infer the concrete type from the parameters.
            dtype = infer_from_params(parsed.params)
            if dtype is None:
                raise TypeError(
                    f"Cannot infer an astropy type from parameters "
                    f"'{parsed.params}' in '{string}'."
                )
            return dtype

        try:
            spec = get_by_selector(parsed.selector)
        except KeyError:
            raise TypeError(
                f"Unknown astropy type selector '{parsed.selector}' in '{string}'."
            ) from None
        return spec.parse_params(parsed.params)

    def construct_array_type(self) -> type[ExtensionArray]:
        return AstropyExtensionArray


class AstropyExtensionArray(ExtensionArray):
    """
    Base class for astropy-backed pandas extension arrays.

    Concrete subclasses (e.g. ``QuantityExtensionArray``) implement the actual
    storage and override ``_from_sequence``/``_from_astropy``/``to_astropy``.
    The base class itself is never instantiated; it only serves as the umbrella
    array type that dispatches the bare-``astropy`` inference path.

    Examples
    --------
    >>> import pandas as pd
    >>> import astropy.units as u
    >>> import pandas_units_extension
    >>> s = pd.Series([1 * u.m, 2 * u.m], dtype="astropy")  # umbrella dispatch
    >>> type(s.array).__name__
    'QuantityExtensionArray'
    """

    @classmethod
    def _from_sequence(
        cls, scalars, *, dtype=None, copy: bool = False
    ) -> "AstropyExtensionArray":
        if cls is not AstropyExtensionArray:
            # A concrete subclass that forgot to override _from_sequence.
            raise NotImplementedError(
                f"{cls.__name__} must implement '_from_sequence'."
            )

        # Umbrella path (bare "astropy"): infer the concrete type from the data.
        spec = _infer_spec_from_scalars(scalars)
        if spec is None:
            raise TypeError(
                "Cannot infer astropy type from data. Write an explicit dtype "
                "such as 'astropy{quantity}[m]' or use 'from_astropy'."
            )
        return spec.array_cls._from_sequence(scalars, dtype=None, copy=copy)

    @classmethod
    def _from_astropy(cls, obj: Any) -> "AstropyExtensionArray":
        """Build the array from a native astropy object. Overridden by subclasses."""
        raise NotImplementedError(f"{cls.__name__} must implement '_from_astropy'.")

    def to_astropy(self) -> Any:
        """Return the native astropy object backing this array. Overridden by subclasses."""
        raise NotImplementedError(f"{type(self).__name__} must implement 'to_astropy'.")


def _infer_spec_from_scalars(scalars):
    """Find the registry spec matching the (first non-null) scalar of ``scalars``."""
    spec = infer_from_object(scalars)
    if spec is not None:
        return spec
    try:
        iterator = iter(scalars)
    except TypeError:
        return None
    for item in iterator:
        spec = infer_from_object(item)
        if spec is not None:
            return spec
    return None


def from_astropy(obj: Any) -> AstropyExtensionArray:
    """
    Build the appropriate pandas extension array from a native astropy object.

    Parameters
    ----------
    obj : object
        A native astropy object (e.g. a :class:`~astropy.units.Quantity`).

    Returns
    -------
    AstropyExtensionArray
        The concrete extension array registered for ``type(obj)``.

    Raises
    ------
    TypeError
        If no registered astropy type handles ``obj``.

    Examples
    --------
    >>> import astropy.units as u
    >>> from pandas_units_extension import from_astropy
    >>> from_astropy(u.Quantity([1, 2, 3], "m"))
    <QuantityExtensionArray>
    [1.0 m, 2.0 m, 3.0 m]
    Length: 3, dtype: astropy{quantity}[m]
    """
    spec = infer_from_object(obj)
    if spec is None:
        raise TypeError(
            f"No registered astropy extension type handles {type(obj).__name__}."
        )
    return spec.array_cls._from_astropy(obj)


@register_series_accessor("astropy")
class AstropySeriesAccessor:
    """
    Umbrella ``.astropy`` accessor for Series, dispatching by array type.

    Accessing ``series.astropy`` returns the concrete accessor subclass
    registered for the series' array (e.g. ``QuantitySeriesAccessor``). Common
    methods such as :meth:`to_astropy` live here; type-specific methods live on
    the subclasses and are simply absent (``AttributeError``) for other types.

    Examples
    --------
    >>> import pandas as pd
    >>> import pandas_units_extension
    >>> s = pd.Series([1, 2, 3], dtype="astropy{quantity}[m]")
    >>> s.astropy.to_astropy()
    <Quantity [1., 2., 3.] m>
    >>> s.astropy.to("mm")
    0    1000.0 mm
    1    2000.0 mm
    2    3000.0 mm
    dtype: astropy{quantity}[mm]
    """

    series: pd.Series

    def __new__(cls, series: pd.Series) -> "AstropySeriesAccessor":
        if cls is AstropySeriesAccessor:
            spec = infer_from_array(series.array)
            if spec is None or spec.series_accessor_cls is None:
                raise AttributeError(
                    "Only astropy-backed Series have the 'astropy' accessor."
                )
            return object.__new__(spec.series_accessor_cls)
        return object.__new__(cls)

    def __init__(self, series: pd.Series) -> None:
        self.series = series

    @property
    def _array(self) -> AstropyExtensionArray:
        """Shortcut to the extension array of the series."""
        return self.series.array  # type: ignore[return-value]

    def to_astropy(self) -> Any:
        """
        Convert the series to its native astropy object.

        Returns
        -------
        object
            The native astropy object (e.g. a ``Quantity``).
        """
        return self._array.to_astropy()


@register_dataframe_accessor("astropy")
class AstropyDataFrameAccessor:
    """
    Umbrella ``.astropy`` accessor for DataFrames.

    Parameters
    ----------
    df : DataFrame
        The dataframe the accessor methods are applied on.

    Examples
    --------
    >>> import pandas as pd
    >>> import pandas_units_extension
    >>> df = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="astropy{quantity}[km]")})
    >>> df.astropy.to_si()
              a
    0  1000.0 m
    1  2000.0 m
    2  3000.0 m
    >>> df.astropy.to_table()["a"].unit
    Unit("km")
    """

    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def to_table(self):
        """
        Convert the DataFrame to an astropy :class:`~astropy.table.QTable`.

        Each astropy-backed column contributes its native object (preserving
        units etc.); other columns contribute their raw values. ``QTable`` is
        used rather than :meth:`Table.from_pandas`, which drops units.

        Returns
        -------
        QTable
            A table with one column per DataFrame column.
        """
        from astropy.table import QTable

        columns: dict[Any, Any] = {}
        for name in self.df.columns:
            col = self.df[name]
            try:
                columns[name] = col.astropy.to_astropy()
            except AttributeError:
                columns[name] = col.to_numpy()
        return QTable(columns)

    def to_si(self) -> pd.DataFrame:
        """
        Convert all astropy-backed columns that support it to SI units.

        Returns
        -------
        DataFrame
            A new DataFrame with convertible columns expressed in SI units.
        """

        def _f(col):
            try:
                return col.astropy.to_si()
            except AttributeError:
                return col

        return self.df.apply(_f)
