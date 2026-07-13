"""
The ``angle`` astropy type: pandas extension for :class:`astropy.coordinates.Angle`.

``Angle`` is a :class:`~astropy.units.Quantity` subclass restricted to angular
units, with extra methods (``wrap_at``, ``dms``/``hms``, ``is_within_bounds``).
This module specialises the ``quantity`` classes accordingly and registers the
``angle`` type so ``astropy{angle}[deg]`` / ``from_astropy(Angle(...))`` resolve
to it.
"""

import astropy.units as u
import pandas as pd
from astropy.coordinates import Angle

from .quantity import (
    QuantityDtype,
    QuantityExtensionArray,
    QuantitySeriesAccessor,
    UnitInstance,
)
from .registry import AstropyTypeSpec, register_astropy_type


def _check_angular(unit: UnitInstance | None) -> None:
    """Raise if ``unit`` is neither ``None`` nor an angular unit."""
    if unit is not None and unit.physical_type != "angle":
        raise ValueError(
            f"AngleDtype requires an angular unit, got '{unit}' "
            f"(physical type '{unit.physical_type}')."
        )


class AngleDtype(QuantityDtype):
    """
    Description of the angle type (``astropy{angle}[<angular unit>]``).

    Like :class:`QuantityDtype` but restricted to angular units.

    Parameters
    ----------
    unit : UnitInstance, optional
        The (angular) physical unit of this dtype.

    Examples
    --------
    >>> import astropy.units as u
    >>> from pandas_units_extension import AngleDtype
    >>> AngleDtype(u.deg)
    AngleDtype("deg")
    >>> AngleDtype(u.deg).name
    'astropy{angle}[deg]'
    >>> AngleDtype(u.m)
    Traceback (most recent call last):
        ...
    ValueError: AngleDtype requires an angular unit, got 'm' (physical type 'length').
    """

    type: type = Angle
    _selector: str = "angle"

    def __init__(self, unit: UnitInstance | None = None) -> None:
        _check_angular(unit)
        super().__init__(unit)

    def construct_array_type(self) -> type[QuantityExtensionArray]:
        return AngleExtensionArray


class AngleExtensionArray(QuantityExtensionArray):
    """
    Pandas extension array for astropy :class:`~astropy.coordinates.Angle` values.

    Behaves like :class:`QuantityExtensionArray` but is restricted to angular
    units, converts to :class:`~astropy.coordinates.Angle` via ``to_astropy`` and
    exposes ``wrap_at``. Arithmetic whose result leaves the angular domain (e.g.
    ``angle / angle``) degrades to a plain :class:`QuantityExtensionArray`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from pandas_units_extension import AngleExtensionArray
    >>> arr = AngleExtensionArray([350, 10], u.deg)
    >>> arr.to_astropy()
    <Angle [350.,  10.] deg>
    >>> arr.wrap_at(180 * u.deg).to_astropy()
    <Angle [-10.,  10.] deg>
    """

    _dtype_cls: "type[AngleDtype]" = AngleDtype

    def _wrap_result(self, quantity: u.Quantity) -> QuantityExtensionArray:
        # Stay an Angle only while the result is still angular; otherwise degrade.
        if quantity.unit.physical_type == "angle":
            return AngleExtensionArray(quantity)
        return QuantityExtensionArray(quantity)

    def to_astropy(self) -> Angle:
        """
        Convert to the native astropy object (an :class:`~astropy.coordinates.Angle`).

        Returns
        -------
        Angle
        """
        return Angle(self.to_quantity())

    def wrap_at(self, wrap_angle: "u.Quantity | str") -> "AngleExtensionArray":
        """
        Wrap the angles into the range ``[wrap_angle - 360 deg, wrap_angle)``.

        Parameters
        ----------
        wrap_angle : Quantity or str
            The wrap angle (e.g. ``180 * u.deg`` or ``"180d"``).

        Returns
        -------
        AngleExtensionArray
        """
        return AngleExtensionArray(self.to_astropy().wrap_at(wrap_angle))


class AngleSeriesAccessor(QuantitySeriesAccessor):
    """
    Accessor adding angle functionality to a Series (``series.astropy``).

    Surfaced when the Series is backed by an :class:`AngleExtensionArray`. Adds
    ``wrap_at``, the sexagesimal decompositions ``dms``/``hms``/``signed_dms``
    (as DataFrames), and ``is_within_bounds`` on top of the inherited quantity
    methods (``to_astropy`` -> ``Angle``, ``to``, ``to_si``, ``unit``).

    Examples
    --------
    >>> import astropy.units as u
    >>> import pandas as pd
    >>> import pandas_units_extension
    >>> s = pd.Series([350, 10], dtype="astropy{angle}[deg]")
    >>> s.astropy.wrap_at(180 * u.deg)
    0    -10.0 deg
    1     10.0 deg
    dtype: astropy{angle}[deg]
    >>> pd.Series([10.5], dtype="astropy{angle}[deg]").astropy.dms
          d     m    s
    0  10.0  30.0  0.0
    """

    def wrap_at(self, wrap_angle: "u.Quantity | str") -> pd.Series:
        """
        Convert to a Series of angles wrapped at ``wrap_angle``.

        Parameters
        ----------
        wrap_angle : Quantity or str
            The wrap angle (e.g. ``180 * u.deg``).

        Returns
        -------
        Series
        """
        return self._wrap(self._array.wrap_at(wrap_angle))

    def _sexagesimal_frame(self, decomposition) -> pd.DataFrame:
        """Turn a per-element ``(d, m, s)``-style namedtuple into a DataFrame."""
        return pd.DataFrame(
            {field: getattr(decomposition, field) for field in decomposition._fields},
            index=self.series.index,
        )

    @property
    def dms(self) -> pd.DataFrame:
        """Sexagesimal degrees as a DataFrame with columns ``d``, ``m``, ``s``."""
        return self._sexagesimal_frame(self._array.to_astropy().dms)

    @property
    def hms(self) -> pd.DataFrame:
        """Sexagesimal hours as a DataFrame with columns ``h``, ``m``, ``s``."""
        return self._sexagesimal_frame(self._array.to_astropy().hms)

    @property
    def signed_dms(self) -> pd.DataFrame:
        """
        Signed sexagesimal degrees as a DataFrame.

        Columns are ``sign``, ``d``, ``m``, ``s`` with non-negative ``d``/``m``/``s``.
        """
        return self._sexagesimal_frame(self._array.to_astropy().signed_dms)

    def is_within_bounds(
        self,
        lower: "u.Quantity | str | None" = None,
        upper: "u.Quantity | str | None" = None,
    ) -> bool:
        """
        Whether all angles are within the given bounds.

        Parameters
        ----------
        lower, upper : Quantity or str or None
            The (inclusive lower / exclusive upper) bounds; ``None`` disables that
            side of the check.

        Returns
        -------
        bool
            ``True`` if every angle is within the bounds.
        """
        return self._array.to_astropy().is_within_bounds(lower, upper)


register_astropy_type(
    AstropyTypeSpec(
        selector="angle",
        native_type=Angle,
        dtype_cls=AngleDtype,
        array_cls=AngleExtensionArray,
        series_accessor_cls=AngleSeriesAccessor,
        parse_params=AngleDtype._parse_params,
    )
)
