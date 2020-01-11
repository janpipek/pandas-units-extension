"""Extension unit and array for pandas.

It is based on astropy Quantities.

Examples
--------
    >>> pd.Series([1, 2, 3], dtype="unit[m]")
    0   1.0 m
    1   2.0 m
    2   3.0 m
    dtype: unit[m]

    >>> pd.Series([123, 21], dtype="unit[km]") / pd.Series([1, 3], dtype="unit[hour]")
    0   123.0 km / h
    1     7.0 km / h
    dtype: unit[km / h]
"""
__all__ = [
    "as_quantity",
    "convert",
    "UnitsDtype",
    "UnitsExtensionArray",
    "UnitsSeriesAccessor",
    "UnitsDataFrameAccessor",
    "Unit",
    "__version__",
]

__version__ = "0.1.0"

from .units import (
    UnitsDtype,
    as_quantity,
    convert,
    UnitsExtensionArray,
    Unit,
    UnitsDataFrameAccessor,
    UnitsSeriesAccessor,
    UnitsDtype,
)
