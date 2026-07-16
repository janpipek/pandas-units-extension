"""Tests for the ``angle`` astropy type (astropy.coordinates.Angle support)."""

import astropy.units as u
import pandas as pd
import pytest
from astropy.coordinates import Angle

from pandas_units_extension import (
    AngleDtype,
    AngleExtensionArray,
    AngleSeriesAccessor,
    QuantityExtensionArray,
    from_astropy,
)
from pandas_units_extension.registry import infer_from_array, infer_from_object


class TestAngleDtype:
    def test_name(self):
        assert AngleDtype(u.deg).name == "astropy{angle}[deg]"
        assert AngleDtype().name == "astropy{angle}"

    def test_repr(self):
        assert repr(AngleDtype(u.deg)) == 'AngleDtype("deg")'

    def test_requires_angular_unit(self):
        with pytest.raises(ValueError, match="requires an angular unit"):
            AngleDtype(u.m)

    @pytest.mark.parametrize("unit", [u.deg, u.rad, u.arcsec, u.hourangle])
    def test_accepts_angular_units(self, unit):
        assert AngleDtype(unit).unit == unit

    def test_construct_from_string(self):
        from pandas_units_extension import AstropyDtype

        dtype = AstropyDtype.construct_from_string("astropy{angle}[deg]")
        assert isinstance(dtype, AngleDtype)
        assert dtype.unit == u.deg

    def test_construct_from_string_non_angular_raises(self):
        from pandas_units_extension import AstropyDtype

        with pytest.raises(ValueError, match="requires an angular unit"):
            AstropyDtype.construct_from_string("astropy{angle}[m]")


class TestAngleInference:
    def test_object_inference_prefers_angle(self):
        # Angle is a Quantity subclass; the most-specific spec must win.
        assert infer_from_object(Angle(1, "deg")).selector == "angle"
        assert infer_from_object(1 * u.deg).selector == "quantity"

    def test_array_inference_prefers_angle(self):
        assert infer_from_array(AngleExtensionArray([1, 2], u.deg)).selector == "angle"
        assert (
            infer_from_array(QuantityExtensionArray([1, 2], u.deg)).selector
            == "quantity"
        )

    def test_series_infers_angle_from_data(self):
        s = pd.Series([Angle(1, "deg"), Angle(2, "deg")], dtype="astropy")
        assert isinstance(s.array, AngleExtensionArray)

    def test_from_astropy(self):
        arr = from_astropy(Angle([1, 2, 3], "deg"))
        assert isinstance(arr, AngleExtensionArray)
        assert arr._unit == u.deg

    def test_selectorless_params_stays_quantity(self):
        # "astropy[deg]" infers by params in registration order -> the base type.
        from pandas_units_extension import AstropyDtype

        dtype = AstropyDtype.construct_from_string("astropy[deg]")
        assert type(dtype).__name__ == "QuantityDtype"


class TestAngleArray:
    def test_series_construction(self):
        s = pd.Series([350, 10, 190], dtype="astropy{angle}[deg]")
        assert isinstance(s.array, AngleExtensionArray)
        assert s.dtype.unit == u.deg

    def test_to_astropy_returns_angle(self):
        s = pd.Series([1, 2, 3], dtype="astropy{angle}[deg]")
        result = s.astropy.to_astropy()
        assert isinstance(result, Angle)
        assert (result == Angle([1, 2, 3], u.deg)).all()

    def test_slice_and_copy_preserve_type(self):
        arr = AngleExtensionArray([1, 2, 3], u.deg)
        assert isinstance(arr[:2], AngleExtensionArray)
        assert isinstance(arr.copy(), AngleExtensionArray)
        assert isinstance(arr.take([0, 2]), AngleExtensionArray)

    def test_to_preserves_angle(self):
        s = pd.Series([1], dtype="astropy{angle}[deg]")
        assert isinstance(s.astropy.to("arcsec").array, AngleExtensionArray)

    def test_addition_stays_angle(self):
        s = pd.Series([10, 20], dtype="astropy{angle}[deg]")
        assert isinstance((s + s).array, AngleExtensionArray)

    def test_ratio_degrades_to_quantity(self):
        s = pd.Series([10, 20], dtype="astropy{angle}[deg]")
        result = (s / s).array
        assert isinstance(result, QuantityExtensionArray)
        assert not isinstance(result, AngleExtensionArray)

    def test_product_degrades_to_quantity(self):
        s = pd.Series([10, 20], dtype="astropy{angle}[deg]")
        assert not isinstance((s * s).array, AngleExtensionArray)


class TestAngleAccessor:
    def test_dispatches_to_angle_accessor(self):
        s = pd.Series([1, 2, 3], dtype="astropy{angle}[deg]")
        assert isinstance(s.astropy, AngleSeriesAccessor)

    def test_wrap_at(self):
        s = pd.Series([350, 10, 190], dtype="astropy{angle}[deg]")
        result = s.astropy.wrap_at(180 * u.deg)
        expected = pd.Series([-10, 10, -170], dtype="astropy{angle}[deg]")
        pd.testing.assert_series_equal(result, expected)

    def test_wrap_at_returns_angle_backed_series(self):
        s = pd.Series([350], dtype="astropy{angle}[deg]")
        assert isinstance(s.astropy.wrap_at(180 * u.deg).array, AngleExtensionArray)

    def test_dms(self):
        s = pd.Series([10.5], dtype="astropy{angle}[deg]")
        result = s.astropy.dms
        assert list(result.columns) == ["d", "m", "s"]
        assert result.loc[0, "d"] == 10.0
        assert result.loc[0, "m"] == 30.0

    def test_hms(self):
        s = pd.Series([180], dtype="astropy{angle}[deg]")
        result = s.astropy.hms
        assert list(result.columns) == ["h", "m", "s"]
        assert result.loc[0, "h"] == 12.0

    def test_signed_dms(self):
        s = pd.Series([-20.25], dtype="astropy{angle}[deg]")
        result = s.astropy.signed_dms
        assert list(result.columns) == ["sign", "d", "m", "s"]
        assert result.loc[0, "sign"] == -1.0
        assert result.loc[0, "d"] == 20.0

    def test_dms_aligns_index(self):
        s = pd.Series([10.5, 20.5], index=["a", "b"], dtype="astropy{angle}[deg]")
        assert list(s.astropy.dms.index) == ["a", "b"]

    def test_is_within_bounds(self):
        s = pd.Series([10, 20, 30], dtype="astropy{angle}[deg]")
        assert s.astropy.is_within_bounds(0 * u.deg, 90 * u.deg) is True
        assert s.astropy.is_within_bounds(0 * u.deg, 15 * u.deg) is False

    def test_inherited_quantity_methods(self):
        s = pd.Series([1], dtype="astropy{angle}[deg]")
        assert s.astropy.unit == u.deg
        assert isinstance(s.astropy.to_astropy(), Angle)

    def test_quantity_series_has_no_angle_methods(self):
        # wrap_at must be absent on a plain quantity-backed Series.
        s = pd.Series([1, 2, 3], dtype="astropy{quantity}[m]")
        assert not hasattr(s.astropy, "wrap_at")
