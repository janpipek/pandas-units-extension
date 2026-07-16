"""
Extension pandas dtypes and arrays for astropy objects.

The package exposes an ``astropy`` umbrella dtype and accessor. Concrete types
are selected with the ``astropy{<type>}[<params>]`` grammar; the available types
are ``quantity`` (an astropy :class:`~astropy.units.Quantity`) and ``angle`` (an
astropy :class:`~astropy.coordinates.Angle`). The selector may be omitted
(``astropy[<params>]``) when the type can be inferred from the parameters.

Examples
--------
    >>> import astropy.units as u
    >>> import pandas as pd
    >>> import pandas_units_extension  # registers the dtypes and accessors
    >>> pd.Series([1, 2, 3], dtype="astropy{quantity}[m]")
    0    1.0 m
    1    2.0 m
    2    3.0 m
    dtype: astropy{quantity}[m]

    >>> (
    ...     pd.Series([123, 21], dtype="astropy{quantity}[km]")
    ...     / pd.Series([1, 3], dtype="astropy{quantity}[hour]")
    ... )
    0    123.0 km / h
    1      7.0 km / h
    dtype: astropy{quantity}[km / h]

    >>> pd.Series([1 * u.m, 2 * u.m], dtype="astropy")  # type inferred from data
    0    1.0 m
    1    2.0 m
    dtype: astropy{quantity}[m]
"""

__all__ = [
    "AstropyDtype",
    "AstropyExtensionArray",
    "AstropyDataFrameAccessor",
    "QuantityDtype",
    "QuantityExtensionArray",
    "QuantitySeriesAccessor",
    "AngleDtype",
    "AngleExtensionArray",
    "AngleSeriesAccessor",
    "from_astropy",
    "InvalidUnitError",
    "InvalidUnitConversionError",
    "__version__",
]

from .base import (
    AstropyDataFrameAccessor,
    AstropyDtype,
    AstropyExtensionArray,
    from_astropy,
)

# Importing the concrete types registers them with the registry.
from .quantity import (
    InvalidUnitConversionError,
    InvalidUnitError,
    QuantityDtype,
    QuantityExtensionArray,
    QuantitySeriesAccessor,
)
from .angle import (
    AngleDtype,
    AngleExtensionArray,
    AngleSeriesAccessor,
)
from .version import __version__
