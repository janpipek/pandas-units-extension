# ruff: disable[F401,F811]
from __future__ import annotations

import operator

import astropy.units as u
import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from pandas.core import ops
from pandas.tests.extension import base
from pandas.tests.extension.base import BaseOpsUtil
from pandas.tests.extension.base.base import BaseExtensionTests
from pandas.tests.extension.conftest import (
    as_array,
    as_frame,
    as_series,
    fillna_method,
    groupby_apply_op,
    invalid_scalar,
    use_numpy,
)

from pandas_units_extension.units import (
    UnitsDtype,
    UnitsExtensionArray,
    UnitsSeriesAccessor,
    InvalidUnitConversion,
)

_all_arithmetic_operators: list[str] = [
    "__add__",  # '__radd__',
    "__sub__",  # '__rsub__',
    "__mul__",  # '__rmul__',
    "__floordiv__",  #'__rfloordiv__',
    "__truediv__",  #'__rtruediv__',
    # '__pow__', # '__rpow__',
    "__mod__",  # '__rmod__'
]


@pytest.fixture(params=_all_arithmetic_operators)
def all_arithmetic_operators(request):
    return request.param


@pytest.fixture(params=[True, False])
def using_nan_is_na(request):
    opt = request.param
    with pd.option_context("future.distinguish_nan_and_na", not opt):
        yield opt


@pytest.fixture
def data():
    return UnitsExtensionArray([1, 2] + 8 * [3], u.m)


@pytest.fixture()
def data_for_twos():
    return UnitsExtensionArray([2] * 10, u.m)


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return UnitsExtensionArray([np.nan, 1] * u.m)


@pytest.fixture
def simple_data():
    return UnitsExtensionArray([1, 2, 3], u.m)


@pytest.fixture
def incoercible_data():
    return [u.Quantity(1, "kg"), u.Quantity(1, "m")]


@pytest.fixture
def coercible_data():
    return [u.Quantity(1, "kg"), u.Quantity(1, "g")]


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def dtype():
    return UnitsDtype(u.m)


@pytest.fixture
def na_cmp():
    # Note: np.nan != np.nan
    def cmp(x, y):
        if np.isnan(x.value):
            return np.isnan(y.value)
        else:
            return x == y

    return cmp


@pytest.fixture
def na_value():
    # Must be the same unit as others
    return np.nan * u.m


@pytest.fixture
def data_for_grouping():
    return UnitsExtensionArray([2, 2, np.nan, np.nan, 1, 1, 2, 3], u.g)


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return UnitsExtensionArray([2, 3, 1], u.m)


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return UnitsExtensionArray([3, np.nan, 1], u.m)


@pytest.fixture
def data_repeated(data):
    def gen(count):
        for _ in range(count):
            yield data

    return gen


# prod forbidden
# kurt, skew not implemented
_all_numeric_reductions = ["sum", "max", "min", "mean", "std", "var", "median"]
#'kurt', 'skew']


@pytest.fixture
def sort_by_key():
    return None


@pytest.fixture(
    params=[
        operator.eq,
        operator.ne,
        operator.le,
        operator.lt,
        operator.ge,
        operator.gt,
    ]
)
def comparison_op(request):
    return request.param


@pytest.fixture(params=_all_numeric_reductions)
def all_numeric_reductions(request):
    return request.param


# These are not implemented
_all_boolean_reductions = ["all", "any"]


@pytest.fixture(params=_all_boolean_reductions)
def all_boolean_reductions(request):
    return request.param


@pytest.fixture(params=[True, False])
def box_in_series(request):
    """Whether to box the data in a Series"""
    return request.param


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestCasting(base.BaseCastingTests):
    def test_compatible_conversion(self):
        s = pd.Series([3, 4], dtype="unit[m]")
        result = s.astype("unit[cm]")
        expected = pd.Series([300, 400], dtype="unit[cm]")
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("generic", [False, True])
    def test_convert_from_object(self, generic):
        s = pd.Series([2, 3] * u.m)
        dtype = "unit" if generic else "unit[m]"
        result = s.astype(dtype)
        expected = pd.Series([2, 3], dtype="unit[m]")
        tm.assert_series_equal(result, expected)

    def test_convert_from_timedelta(self):
        s = pd.Series(pd.timedelta_range(0, periods=3, freq="h"))
        result = s.astype("unit")
        expected = pd.Series([0, 3600, 7200], dtype="unit[s]")
        tm.assert_series_equal(result, expected)

    def test_astype_timedelta(self):
        s = pd.Series([0, 1, 2], dtype="unit[h]")
        result = s.astype("timedelta64[ns]")
        expected = pd.Series(pd.timedelta_range(0, periods=3, freq="h"))
        tm.assert_series_equal(result, expected)


class TestDtype(base.BaseDtypeTests):
    pass


class TestGroupBy(base.BaseGroupbyTests):
    @pytest.mark.xfail(
        pd.__version__ < "3.1.0",
        reason="Test fails on pandas below 3.1.0, see pandas GH #64111",
    )
    def test_groupby_agg_extension(self, data_for_grouping):
        return super().test_groupby_agg_extension(data_for_grouping)

    @pytest.mark.xfail(
        reason="The UnitsDtype is non of the dtypes for which the test expects a successful grouping, therefore no TypeError is raised",
    )
    def test_in_numeric_groupby(self, data_for_grouping):
        return super().test_in_numeric_groupby(data_for_grouping)


class TestGetitem(base.BaseGetitemTests):
    def test_unitless(self):
        series = pd.Series([0, 1, 2], dtype="unit[]")
        new_index = [2, 4]
        result = series.reindex(new_index)
        expected = pd.Series([2, np.nan], dtype="unit[]", index=new_index)
        tm.assert_series_equal(result, expected)


class TestInterface(base.BaseInterfaceTests):
    def test_contains_unit_aware_na_values(self, data_missing):
        """Test that various na-values are or are not in data_missing"""
        # data_missing is of dtype unit[m] so `np.nan * u.m` should be in it
        assert (np.nan * u.m) in data_missing

        # UnitsExtensionArray is flexible in regards to the unit of a na-value
        # for __contains__() as long as the physical type is the same (here length),
        # so `np.nan * u.cm` should also be in data_missing
        assert (np.nan * u.cm) in data_missing

        # However a different physical type like time for `np.nan * u.s` should not be
        assert (np.nan * u.s) not in data_missing


class TestMethods(base.BaseMethodsTests):
    def test_searchsorted_unit_aware(self, data_for_sorting, as_series):
        arr: UnitsExtensionArray = UnitsExtensionArray([1, 2, 3], u.m)

        if as_series:
            arr = pd.Series(arr)

        # Check that simple 1 m equivalent is same position as first element,
        # therefore 0 for left and 1 for right side
        a = u.Quantity(1, "m")
        assert arr.searchsorted(a) == 0
        assert arr.searchsorted(a, side="right") == 1

        # Check that 200 cm is equivalent to 2 m in searchsorted
        b = u.Quantity(200, "cm")
        assert arr.searchsorted(b) == 1
        assert arr.searchsorted(b, side="right") == 2

        # Check that 0.003 km is equivalent to 3 m in searchsorted
        c = u.Quantity(0.003, "km")
        assert arr.searchsorted(c) == 2
        assert arr.searchsorted(c, side="right") == 3


class TestReshaping(base.BaseReshapingTests):
    pass


class TestReduce(base.BaseReduceTests):
    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # List all supported numeric reductions
        return op_name in {"sum", "max", "min", "mean", "std", "var", "median"}

    def _get_expected_reduction_dtype(self, arr, op_name: str, skipna: bool):
        # Besides `var` all reductions retain the same unit so same dtype.
        # However `var` returns a squared unit and the new expected dtype is calculated and returned
        if op_name in {"var"}:
            return UnitsDtype(arr.unit**2)
        return arr.dtype

    def check_reduce(self, ser: pd.Series, op_name: str, skipna: bool):
        # We must check float values
        result = getattr(ser, op_name)(skipna=skipna).value
        expected = getattr(ser.astype("float64"), op_name)(skipna=skipna)
        np.testing.assert_almost_equal(result, expected)

    # We include some trusted results on top of pandas' ones
    def test_sum(self, data, data_missing):
        assert pd.Series(data).sum() == 27 * u.m
        assert np.isnan(pd.Series(data_missing).sum(skipna=False))
        assert pd.Series(data_missing).sum() == 1 * u.m

    def test_mean(self, data):
        assert np.allclose(pd.Series(data).mean() / u.m, 2.7)

    def test_min(self, data):
        assert pd.Series(data).min() == 1 * u.m

    def test_max(self, data):
        assert pd.Series(data).max() == 3 * u.m

    def test_median(self, data):
        assert pd.Series(data).median() == 3 * u.m

    def test_std(self, data):
        assert np.allclose(pd.Series(data).std() / u.m, 0.6749486)

    def test_sem(self, data):
        assert np.allclose(pd.Series(data).sem() / u.m, 0.21343747458109494)

    def test_var(self, data):
        assert np.allclose(pd.Series(data).var() / (u.m**2), 0.4555555555555555)

    def test_unsupported(self, data):
        for method in ["any", "all", "prod"]:
            with pytest.raises(TypeError):
                getattr(pd.Series(data), method)()


class TestSetitem(base.BaseSetitemTests):
    @pytest.mark.xfail(
        reason="The `to_numpy` function used in this test wrongfully assumes a view and therefore sets the writable flag to false."
    )
    def test_readonly_propagates_to_numpy_array_method(self, data):
        super().test_readonly_propagates_to_numpy_array_method(data)

    @pytest.mark.xfail(
        reason="Index.get_loc() returns a InvalidIndexError instead of the expected KeyError (key not in index for setter) because Quantity is an instance of abc.Iterable"
    )
    def test_loc_setitem_with_expansion_preserves_ea_index_dtype(self, data):
        super().test_loc_setitem_with_expansion_preserves_ea_index_dtype(data)


class TestParsing(base.BaseParsingTests):
    @pytest.mark.parametrize("generic", [False, True])
    def test_series_from_string_list(self, generic):
        source = ["1 m", "2 m"]
        dtype = "unit" if generic else "unit[m]"
        result = pd.Series(source, dtype=dtype)
        expected = pd.Series([1, 2], dtype="unit[m]")
        tm.assert_series_equal(result, expected)


class TestMissing(base.BaseMissingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestQuantile:
    # Regression test for issue #1
    def test_returns_correct_unit(self):
        source = pd.Series([1, 2, 3], dtype="unit[m]")
        result = source.quantile(0.5)
        assert result == 2.0 * u.m


class TestArithmeticsOps(base.BaseArithmeticOpsTests):
    divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None

    def test_arith_series_with_scalar_pow(self, data):
        s = pd.Series(data)
        result = s**2
        expected = pd.Series([1, 4] + 8 * [9], dtype="unit[m^2]")
        tm.assert_series_equal(result, expected)

        result2 = s ** (-2)
        expected2 = pd.Series([1, 1 / 4] + 8 * [1 / 9], dtype="unit[m^(-2)]")
        tm.assert_series_equal(result2, expected2)

    def test_error(self, data, all_arithmetic_operators):
        pass

    def test_add_incompatible_units(self):
        s1 = pd.Series([1, 2, 3, 4], dtype="unit[kg]")
        s2 = pd.Series([3, 4, 3, 4], dtype="unit[m]")
        with pytest.raises(u.UnitConversionError):
            s1 + s2

    def test_add_compatible_units(self):
        s1 = pd.Series([1, 2, 3, 4], dtype="unit[m]")
        s2 = pd.Series([3, 4, 3, 4], dtype="unit[km]")

        expected = pd.Series([3001, 4002, 3003, 4004], dtype="unit[m]")
        result = s1 + s2
        tm.assert_series_equal(expected, result)

    def test_divmod(self, data):
        """Overwritten from base class to compare with Quantity object instead of dimensionless 0"""
        ser = pd.Series(data)
        self._check_divmod_op(ser, divmod, ser[0])
        self._check_divmod_op(ser[0], ops.rdivmod, ser)

    @pytest.mark.xfail(reason="Will be resolved in a future PR.")
    def test_radd(self):
        result = (5 * u.cm) + pd.Series([1, 2, 3], dtype="unit[m]")
        expected = pd.Series([105, 205, 305], dtype="unit[cm]")
        tm.assert_series_equal(result, expected)

    #@pytest.mark.xfail(reason="Makes no sense for pandas-provided fixtures")
    #def test_divmod_series_array(self, data, data_for_twos):
    #    super().test_divmod_series_array(data, data_for_twos)


class TestComparisonOps(base.BaseComparisonOpsTests):
    compare_scalar_mark_xfail: pytest.MarkDecorator = pytest.mark.xfail(
        pd.__version__ < "3.1.0",
        reason="Test fails on pandas below 3.1.0, see pandas GH #64365",
    )

    @pytest.mark.parametrize(
        "comparison_op",
        [
            pytest.param(operator.le, marks=compare_scalar_mark_xfail),
            pytest.param(operator.lt, marks=compare_scalar_mark_xfail),
            pytest.param(operator.ge, marks=compare_scalar_mark_xfail),
            pytest.param(operator.gt, marks=compare_scalar_mark_xfail),
        ],
    )
    def test_compare_scalar(self, data, comparison_op):
        return super().test_compare_scalar(data, comparison_op)

    def test_comparable_units(self):
        s1 = pd.Series([1000, 2000, 3000], dtype="unit[m]")
        s2 = pd.Series([1, 2, 3], dtype="unit[km]")
        s3 = pd.Series([1, 3, 0], dtype="unit[km]")

        assert all(s1 == s2)

        result = s1 < s3
        expected = pd.Series([False, True, False])
        tm.assert_series_equal(expected, result)

    def test_temperature_comparison(self):
        s1 = pd.Series([0, -10, 10], dtype="unit[deg_C]")
        s2 = pd.Series([270, 270, 270], dtype="unit[K]")

        result = s1 < s2
        expected = pd.Series([False, True, False])
        tm.assert_series_equal(expected, result)

    @pytest.mark.parametrize(
        "op",
        [
            operator.le,
            operator.lt,
            operator.ge,
            operator.gt,
        ],
    )
    def test_incomparable_units(self, op):
        s1 = pd.Series([1000, 2000, 3000], dtype="unit[m]")
        s2 = pd.Series([1000, 2000, 3000], dtype="unit[s]")

        with pytest.raises(InvalidUnitConversion):
            op(s1, s2)

    @pytest.mark.parametrize(
        "other",
        [
            pytest.param(1, id="number"),
            pytest.param(pd.Series([1, 2, 3]), id="series"),
        ],
    )
    @pytest.mark.parametrize(
        "op",
        [
            operator.le,
            operator.lt,
            operator.ge,
            operator.gt,
        ],
    )
    def test_with_incompatible_non_units(self, op, other):
        s = pd.Series([1000, 2000, 3000], dtype="unit[m]")
        with pytest.raises(InvalidUnitConversion):
            op(s, other)
        with pytest.raises(InvalidUnitConversion):
            op(other, s)

    @pytest.mark.parametrize(
        ("other", "result"),
        [
            pytest.param(1, pd.Series([False, False]), id="number"),
            pytest.param("1 m", pd.Series([True, False]), id="string-as-unit"),
            pytest.param("m", pd.Series([False, False]), id="string"),
            pytest.param(1 * u.m, pd.Series([True, False]), id="quantity"),
            pytest.param(pd.Series([1, 2]), pd.Series([False, False]), id="series"),
            pytest.param(
                pd.Series([100, 50], dtype="unit[cm]"),
                pd.Series([True, False]),
                id="series-with-compatible-unit",
            ),
            pytest.param(
                pd.Series([1, 2], dtype="unit[kg]"),
                pd.Series([False, False]),
                id="series-with-incompatible-unit",
            ),
        ],
    )
    @pytest.mark.parametrize(
        "op",
        [
            operator.eq,
            operator.ne,
        ],
    )
    def test_eq_ne(self, other, result, op):
        s = pd.Series([1, 2], dtype="unit[m]")
        if op == operator.eq:
            expected = result
        else:
            expected = ~result
        result = op(s, other)
        tm.assert_series_equal(result, expected)


class TestRepr:
    def test_repr(self, simple_data):
        expected: str = (
            "<UnitsExtensionArray>\n[1.0 m, 2.0 m, 3.0 m]\nLength: 3, dtype: unit[m]"
        )
        assert expected == repr(simple_data)

    def test_series_repr(self, simple_data):
        expected: str = "0    1.0 m\n1    2.0 m\n2    3.0 m\ndtype: unit[m]"
        assert expected == repr(pd.Series(simple_data))


class TestUnitsSeriesAccessor(BaseOpsUtil):
    def test_init(self, simple_data):
        s = pd.Series(simple_data)
        assert isinstance(s.units, UnitsSeriesAccessor)

    def test_invalid_type(self):
        s = pd.Series([1, 2, 3])
        with pytest.raises(AttributeError):
            _ = s.units

    def test_to(self, simple_data):
        s = pd.Series(simple_data)
        result = s.units.to("mm")
        expected = pd.Series([1000, 2000, 3000], dtype="unit[mm]")
        tm.assert_series_equal(result, expected)

    def test_to_quantity(self, simple_data):
        s = pd.Series(simple_data)
        result = s.units.to_quantity()
        expected = u.Quantity([1, 2, 3], u.m)
        assert isinstance(result, u.Quantity)
        assert (result == expected).all()

    def test_unit(self, simple_data):
        s = pd.Series(simple_data)
        assert s.units.unit == u.m

    def test_to_si(self):
        s = pd.Series([1, 2, 3], dtype="unit[km]")
        result = s.units.to_si()
        expected = pd.Series([1000, 2000, 3000], dtype="unit[m]")
        tm.assert_series_equal(result, expected)

    def test_to_si_composite_unit(self):
        s = pd.Series([1, 2, 3], dtype="unit[km/h]")
        result = s.units.to_si()
        expected = pd.Series([1000 / 3600, 2000 / 3600, 3000 / 3600], dtype="unit[m/s]")
        tm.assert_series_equal(result, expected)

    def test_temperature(self):
        s = pd.Series([0, 100], dtype="unit[deg_C]")
        s_f = s.units.to("deg_F")
        s_f_expected = pd.Series([32, 212], dtype="unit[deg_F]")
        tm.assert_series_equal(s_f, s_f_expected)


class TestUnitsDataFrameAccessor(BaseOpsUtil):
    def test_df_to_si(self):
        df = pd.DataFrame(
            {
                "a": pd.Series([1, 2, 3], dtype="unit[km]"),
                "b": pd.Series([2, 3, 4], dtype="unit[hour]"),
            }
        )
        result = df.units.to_si()
        expected = pd.DataFrame(
            {
                "a": pd.Series([1000, 2000, 3000], dtype="unit[m]"),
                "b": pd.Series([7200, 10800, 14400], dtype="unit[s]"),
            }
        )
        tm.assert_frame_equal(result, expected)


class TestVarious(BaseExtensionTests):
    def test_concat_compatible(self):
        """Test concatenation of Series with compatible units.

        Both units are of same physical type (length), expected values are converted to first unit, in this case meter.
        """
        s1 = pd.Series(["1 m"], dtype="unit")
        s2 = pd.Series(["1 ft"], dtype="unit")
        concatenated = pd.concat([s1, s2]).reset_index(drop=True)
        expected = pd.Series([1, 0.3048], dtype="unit[m]")
        tm.assert_series_equal(expected, concatenated)

    def test_concat_incompatible(self):
        """Test concatenation of Series with incompatible units.

        Both units are of different physical types (length vs speed), no conversion is done and dtype should be object.
        """
        s1 = pd.Series(["1 m"], dtype="unit")
        s2 = pd.Series(["1 m/s"], dtype="unit")
        concatenated = pd.concat([s1, s2]).reset_index(drop=True)
        expected = pd.Series([u.Quantity("1 m"), u.Quantity("1 m/s")], dtype=object)
        tm.assert_series_equal(expected, concatenated)

    @pytest.mark.xfail(
        pd.__version__ < "3.1.0",
        reason="Test fails on pandas below 3.1.0, see pandas GH #62523",
    )
    def test_add_new_value_with_different_unit(self):
        s1 = pd.Series(["1 m"], dtype="unit")
        s1.at[1] = u.Quantity("1 ft")
        expected = pd.Series(["1.0 m", "0.3048 m"], dtype="unit")
        tm.assert_series_equal(expected, s1)

    def test_set_value_with_different_unit(self):
        s1 = pd.Series(["1 m"], dtype="unit")
        s1[0] = u.Quantity("1 ft")
        expected = pd.Series(["0.3048 m"], dtype="unit")
        tm.assert_series_equal(expected, s1)

    def test_unique(self):
        s = pd.Series([1, np.nan, np.nan], dtype="unit[m]")
        unique = s.unique()
        expected = UnitsExtensionArray([1, np.nan], unit="m")
        assert unique.unit == expected.unit
        np.testing.assert_equal(expected.value, unique.value)
