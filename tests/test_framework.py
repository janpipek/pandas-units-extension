"""Tests for the ``astropy`` framework: grammar, registry, dispatch and ingest."""

import astropy.units as u
import numpy as np
import pandas as pd
import pytest
from astropy.table import QTable

from pandas_units_extension import (
    AstropyDtype,
    QuantityDtype,
    QuantityExtensionArray,
    QuantitySeriesAccessor,
    from_astropy,
)
from pandas_units_extension._grammar import ParsedDtype, parse_dtype_string
from pandas_units_extension.registry import (
    get_by_selector,
    infer_from_array,
    infer_from_object,
    infer_from_params,
)


class TestGrammar:
    @pytest.mark.parametrize(
        ("string", "expected"),
        [
            ("astropy", ParsedDtype(None, None)),
            ("astropy{quantity}", ParsedDtype("quantity", None)),
            ("astropy{quantity}[km/s]", ParsedDtype("quantity", "km/s")),
            ("astropy{quantity}[]", ParsedDtype("quantity", "")),
            ("astropy[km/s]", ParsedDtype(None, "km/s")),
            ("astropy[]", ParsedDtype(None, "")),
            (
                "astropy{representation}[cartesian]",
                ParsedDtype("representation", "cartesian"),
            ),
        ],
    )
    def test_parses(self, string, expected):
        assert parse_dtype_string(string) == expected

    @pytest.mark.parametrize(
        "string", ["astropy{}", "unit", "float64", "astropy_quantity"]
    )
    def test_invalid_raises(self, string):
        with pytest.raises(TypeError):
            parse_dtype_string(string)

    def test_non_string_raises(self):
        with pytest.raises(TypeError):
            parse_dtype_string(0)  # type: ignore[arg-type]


class TestRegistry:
    def test_get_by_selector(self):
        spec = get_by_selector("quantity")
        assert spec.native_type is u.Quantity
        assert spec.array_cls is QuantityExtensionArray
        assert spec.dtype_cls is QuantityDtype

    def test_get_by_selector_unknown(self):
        with pytest.raises(KeyError):
            get_by_selector("nope")

    def test_infer_from_object(self):
        assert infer_from_object(1 * u.m).selector == "quantity"
        assert infer_from_object(u.Quantity([1, 2], "m")).selector == "quantity"

    def test_infer_from_object_no_match(self):
        assert infer_from_object(1.0) is None
        assert infer_from_object("1 m") is None

    def test_infer_from_array(self):
        arr = QuantityExtensionArray([1, 2, 3], u.m)
        assert infer_from_array(arr).selector == "quantity"

    def test_infer_from_array_no_match(self):
        assert infer_from_array(pd.array([1, 2, 3])) is None

    def test_infer_from_params(self):
        dtype = infer_from_params("km/s")
        assert isinstance(dtype, QuantityDtype)
        assert dtype.unit == u.km / u.s

    def test_infer_from_params_no_match(self):
        assert infer_from_params("definitely not a unit") is None


class TestDtypeConstruction:
    @pytest.mark.parametrize(
        ("string", "unit"),
        [
            ("astropy{quantity}[m]", u.m),
            ("astropy{quantity}[km/s]", u.km / u.s),
            ("astropy{quantity}[]", u.Unit("")),
        ],
    )
    def test_typed_string(self, string, unit):
        dtype = AstropyDtype.construct_from_string(string)
        assert isinstance(dtype, QuantityDtype)
        assert dtype.unit == unit

    @pytest.mark.parametrize(
        ("string", "unit"),
        [
            ("astropy[m]", u.m),
            ("astropy[km/s]", u.km / u.s),
            ("astropy[]", u.Unit("")),
        ],
    )
    def test_selectorless_params_string(self, string, unit):
        # "astropy[...]" infers the concrete type from the parameters.
        dtype = AstropyDtype.construct_from_string(string)
        assert isinstance(dtype, QuantityDtype)
        assert dtype.unit == unit

    def test_selectorless_params_unparseable_raises(self):
        with pytest.raises(TypeError, match="Cannot infer an astropy type"):
            AstropyDtype.construct_from_string("astropy[definitely not a unit]")

    def test_bare_quantity_string(self):
        dtype = AstropyDtype.construct_from_string("astropy{quantity}")
        assert isinstance(dtype, QuantityDtype)
        assert dtype.unit is None

    def test_bare_astropy_string(self):
        dtype = AstropyDtype.construct_from_string("astropy")
        assert type(dtype) is AstropyDtype
        assert dtype.name == "astropy"

    def test_unknown_selector_raises(self):
        with pytest.raises(TypeError, match="Unknown astropy type selector"):
            AstropyDtype.construct_from_string("astropy{nope}")

    def test_name_roundtrip(self):
        dtype = QuantityDtype(u.m**2)
        assert dtype.name == "astropy{quantity}[m2]"
        assert QuantityDtype.construct_from_string(dtype.name) == dtype

    def test_repr(self):
        assert repr(QuantityDtype()) == "QuantityDtype()"
        assert repr(QuantityDtype(u.m)) == 'QuantityDtype("m")'


class TestIngest:
    def test_series_typed(self):
        s = pd.Series([1, 2, 3], dtype="astropy{quantity}[km/s]")
        assert isinstance(s.array, QuantityExtensionArray)
        assert s.dtype.unit == u.km / u.s

    def test_series_selectorless_params(self):
        s = pd.Series([1, 2, 3], dtype="astropy[km/s]")
        assert isinstance(s.array, QuantityExtensionArray)
        assert s.dtype.unit == u.km / u.s

    def test_series_bare_infers_from_quantity(self):
        s = pd.Series([1 * u.m, 2 * u.m], dtype="astropy")
        assert isinstance(s.array, QuantityExtensionArray)
        assert s.dtype.unit == u.m

    def test_series_bare_from_quantity_object(self):
        s = pd.Series(u.Quantity([1, 2, 3], "m"), dtype="astropy")
        assert s.dtype.unit == u.m

    def test_series_bare_without_type_info_raises(self):
        with pytest.raises(TypeError, match="Cannot infer astropy type"):
            pd.Series([1.0, 2.0], dtype="astropy")

    def test_pd_array_with_dtype(self):
        arr = pd.array([1 * u.m, 2 * u.m], dtype="astropy")
        assert isinstance(arr, QuantityExtensionArray)
        assert arr._unit == u.m

    def test_from_astropy(self):
        arr = from_astropy(u.Quantity([1, 2, 3], "m"))
        assert isinstance(arr, QuantityExtensionArray)
        assert arr._unit == u.m

    def test_from_astropy_unsupported(self):
        with pytest.raises(TypeError, match="No registered astropy extension type"):
            from_astropy("not an astropy object")


class TestSeriesAccessor:
    def test_dispatches_to_quantity_accessor(self):
        s = pd.Series([1, 2, 3], dtype="astropy{quantity}[m]")
        assert isinstance(s.astropy, QuantitySeriesAccessor)

    def test_to_astropy(self):
        s = pd.Series([1, 2, 3], dtype="astropy{quantity}[m]")
        result = s.astropy.to_astropy()
        assert isinstance(result, u.Quantity)
        assert (result == u.Quantity([1, 2, 3], u.m)).all()

    def test_plain_series_has_no_accessor(self):
        with pytest.raises(AttributeError):
            _ = pd.Series([1, 2, 3]).astropy


class TestDataFrameAccessor:
    def test_to_table_preserves_units(self):
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype="astropy{quantity}[km]"),
                "b": pd.Series([4, 5, 6], dtype="astropy{quantity}[s]"),
            }
        )
        table = df.astropy.to_table()
        assert isinstance(table, QTable)
        assert table["a"].unit == u.km
        assert table["b"].unit == u.s
        np.testing.assert_array_equal(table["a"].value, [1, 2, 3])

    def test_to_table_mixed_columns(self):
        df = pd.DataFrame(
            {
                "q": pd.Series([1, 2, 3], dtype="astropy{quantity}[m]"),
                "plain": [10, 20, 30],
            }
        )
        table = df.astropy.to_table()
        assert table["q"].unit == u.m
        np.testing.assert_array_equal(table["plain"], [10, 20, 30])

    def test_to_si_ignores_non_astropy_columns(self):
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype="astropy{quantity}[km]"),
                "plain": [10, 20, 30],
            }
        )
        result = df.astropy.to_si()
        expected_a = pd.Series([1000, 2000, 3000], dtype="astropy{quantity}[m]")
        pd.testing.assert_series_equal(result["a"], expected_a, check_names=False)
        np.testing.assert_array_equal(result["plain"], [10, 20, 30])
