import numpy as np
import pandas as pd
import pandas.testing as tm
from astropy.units import m
from astropy.units import Quantity
from astropy.units import Unit
from astropy.units import UnitConversionError
from pandas.tests.extension import base
from pandas.tests.extension.base import BaseOpsUtil
from pandas.tests.extension.base.base import BaseExtensionTests
from pandas.tests.extension.conftest import *

from pandas_units_extension.units import UnitsDtype
from pandas_units_extension.units import UnitsExtensionArray
from pandas_units_extension.units import UnitsSeriesAccessor

try:
    from pandas.conftest import all_arithmetic_operators, all_compare_operators
except:
    _all_arithmetic_operators = [
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

    @pytest.fixture(params=["__eq__", "__ne__", "__le__", "__lt__", "__ge__", "__gt__"])
    def all_compare_operators(request):
        return request.param


@pytest.fixture
def data():
    return UnitsExtensionArray([1, 2] + 98 * [3], m)


@pytest.fixture()
def data_for_twos():
    return UnitsExtensionArray([2] * 100, m)


@pytest.fixture
def data_missing():
    """Length-2 array with [NA, Valid]"""
    return UnitsExtensionArray([np.nan * m, 1 * m])


@pytest.fixture
def simple_data():
    return UnitsExtensionArray([1, 2, 3], m)


@pytest.fixture
def incoercible_data():
    return [Quantity(1, "kg"), Quantity(1, "m")]


@pytest.fixture
def coercible_data():
    return [Quantity(1, "kg"), Quantity(1, "g")]


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture giving 'data' and 'data_missing'"""
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture(params=[" ", "mm", "kg s"])
def dtype(request):
    return UnitsDtype(request.param)


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
    return np.nan * m


@pytest.fixture
def data_for_grouping():
    return UnitsExtensionArray([2, 2, np.nan, np.nan, 1, 1, 2, 3], "g")


@pytest.fixture
def data_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return UnitsExtensionArray([2, 3, 1], "m")


@pytest.fixture
def data_missing_for_sorting():
    """Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return UnitsExtensionArray([3, np.nan, 1], "m")


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
        self.assert_series_equal(result, expected)

    @pytest.mark.parametrize("generic", [False, True])
    def test_convert_from_object(self, generic):
        s = pd.Series([2 * m, 3 * m])
        dtype = "unit" if generic else "unit[m]"
        result = s.astype(dtype)
        expected = pd.Series([2, 3], dtype="unit[m]")
        self.assert_series_equal(result, expected)

    def test_convert_from_timedelta(self):
        s = pd.Series(pd.timedelta_range(0, periods=3, freq="h"))
        result = s.astype("unit")
        expected = pd.Series([0, 3600, 7200], dtype="unit[s]")
        self.assert_series_equal(result, expected)

    def test_astype_timedelta(self):
        s = pd.Series([0, 1, 2], dtype="unit[h]")
        result = s.astype("timedelta64[ns]")
        expected = pd.Series(pd.timedelta_range(0, periods=3, freq="h"))
        self.assert_series_equal(result, expected)


class TestDtype(base.BaseDtypeTests):
    pass


class TestGroupBy(base.BaseGroupbyTests):
    pass


class TestGetitem(base.BaseGetitemTests):
    def test_unitless(self):
        series = pd.Series([0, 1, 2], dtype="unit[]")
        new_index = [2, 4]
        result = series.reindex(new_index)
        expected = pd.Series([2, np.nan], dtype="unit[]", index=new_index)
        self.assert_series_equal(result, expected)


class TestInterface(base.BaseInterfaceTests):
    def test_array_interface(self, data):
        # There is no such thing as array of Quantities
        result = np.array(data)
        assert result[0] == data.value[0]

        result = np.array(data.value, dtype=object)
        expected = np.array(list(data.value), dtype=object)
        np.testing.assert_array_equal(result, expected)


class TestMethods(base.BaseMethodsTests):
    # TODO: Report bug, boolean to UnitsExtensionArray requested
    test_combine_le = None

    # TODO: strange results
    test_searchsorted = None


class TestReshaping(base.BaseReshapingTests):
    # TODO: Implement this correctly?
    test_concat_mixed_dtypes = None

    # TODO: np.nan * u.m in the result not expected in the test
    test_unstack = None


class TestBooleanReduce(base.BaseBooleanReduceTests):
    def check_reduce(self, s, op_name, skipna):
        with pytest.raises(TypeError):
            getattr(s, op_name)(skipna=skipna)


class TestNumericReduce(base.BaseNumericReduceTests):
    def check_reduce(self, s, op_name, skipna):
        # We must check float values
        result = getattr(s, op_name)(skipna=skipna).value
        expected = getattr(s.astype("float64"), op_name)(skipna=skipna)
        np.testing.assert_almost_equal(result, expected)

    # We include some trusted results on top of pandas' ones
    def test_sum(self, data, data_missing):
        assert pd.Series(data).sum() == 297 * m
        assert np.isnan(pd.Series(data_missing).sum(skipna=False))
        assert pd.Series(data_missing).sum() == 1 * m

    def test_mean(self, data):
        assert np.allclose(pd.Series(data).mean() / m, 2.97)

    def test_min(self, data):
        assert pd.Series(data).min() == 1 * m

    def test_max(self, data):
        assert pd.Series(data).max() == 3 * m

    def test_median(self, data):
        assert pd.Series(data).median() == 3 * m

    def test_std(self, data):
        assert np.allclose(pd.Series(data).std() / m, 0.2227015)

    def test_sem(self, data):
        assert np.allclose(pd.Series(data).sem() / m, 0.02227015033536137)

    def test_var(self, data):
        assert np.allclose(pd.Series(data).var() / (m ** 2), 0.0495959595959596)

    def test_unsupported(self, data):
        for method in ["any", "all", "prod"]:
            with pytest.raises(TypeError):
                getattr(pd.Series(data), method)()


class TestSetitem(base.BaseSetitemTests):
    # TODO: Report bug?
    test_setitem_mask_broadcast = None


class TestParsing(base.BaseParsingTests):
    @pytest.mark.parametrize("generic", [False, True])
    def test_series_from_string_list(self, generic):
        source = ["1 m", "2 m"]
        dtype = "unit" if generic else "unit[m]"
        result = pd.Series(source, dtype=dtype)
        expected = pd.Series([1, 2], dtype="unit[m]")
        self.assert_series_equal(result, expected)


class TestMissing(base.BaseMissingTests):
    pass


class TestPrinting(base.BasePrintingTests):
    pass


class TestArithmeticsOps(base.BaseArithmeticOpsTests):
    # divmod_exc = None
    series_scalar_exc = None
    frame_scalar_exc = None
    series_array_exc = None

    def test_arith_series_with_scalar_pow(self, data):
        s = pd.Series(data)
        result = s ** 2
        expected = pd.Series([1, 4] + 98 * [9], dtype="unit[m^2]")
        self.assert_series_equal(result, expected)

        result2 = s ** (-2)
        expected2 = pd.Series([1, 1 / 4] + 98 * [1 / 9], dtype="unit[m^(-2)]")
        self.assert_series_equal(result2, expected2)

    def test_error(self, data, all_arithmetic_operators):
        pass

    def test_add_incompatible_units(self):
        s1 = pd.Series([1, 2, 3, 4], dtype="unit[kg]")
        s2 = pd.Series([3, 4, 3, 4], dtype="unit[m]")
        with pytest.raises(UnitConversionError):
            s1 + s2

    def test_add_compatible_units(self):
        s1 = pd.Series([1, 2, 3, 4], dtype="unit[m]")
        s2 = pd.Series([3, 4, 3, 4], dtype="unit[km]")

        expected = pd.Series([3001, 4002, 3003, 4004], dtype="unit[m]")
        result = s1 + s2
        self.assert_series_equal(expected, result)

    @pytest.mark.xfail(reason="Not implemented yet")
    def test_divmod(self, data):
        raise NotImplementedError


class TestComparisonOps(base.BaseComparisonOpsTests):
    def test_comparable_units(self):
        s1 = pd.Series([1000, 2000, 3000], dtype="unit[m]")
        s2 = pd.Series([1, 2, 3], dtype="unit[km]")
        s3 = pd.Series([1, 3, 0], dtype="unit[km]")

        assert all(s1 == s2)

        result = s1 < s3
        expected = pd.Series([False, True, False])
        self.assert_series_equal(expected, result)

    def test_temperature_comparison(self):
        s1 = pd.Series([0, -10, 10], dtype="unit[deg_C]")
        s2 = pd.Series([270, 270, 270], dtype="unit[K]")

        result = s1 < s2
        expected = pd.Series([False, True, False])
        self.assert_series_equal(expected, result)

    def test_incomparable_units(self):
        s1 = pd.Series([1000, 2000, 3000], dtype="unit[m]")
        s2 = pd.Series([1000, 2000, 3000], dtype="unit[s]")

        assert all(s1 != s2)

        with pytest.raises(TypeError):
            s1 < s2


class TestRepr:
    def test_repr(self, simple_data):
        assert (
            "<UnitsExtensionArray>\n[1.0 m, 2.0 m, 3.0 m]\nLength: 3, dtype: unit[m]"
            == repr(simple_data)
        )

    def test_series_repr(self, simple_data):
        assert "0   1.0 m\n1   2.0 m\n2   3.0 m\ndtype: unit[m]" == repr(
            pd.Series(simple_data)
        )


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
        self.assert_series_equal(result, expected)

    def test_unit(self, simple_data):
        s = pd.Series(simple_data)
        assert s.units.unit == Unit("m")

    def test_to_si(self):
        s = pd.Series([1, 2, 3], dtype="unit[km]")
        result = s.units.to_si()
        expected = pd.Series([1000, 2000, 3000], dtype="unit[m]")
        self.assert_series_equal(result, expected)

    def test_temperature(self):
        s = pd.Series([0, 100], dtype="unit[deg_C]")

        s_f = s.units.to("deg_F")
        s_f_expected = pd.Series([32, 212], dtype="unit[deg_F]")
        self.assert_series_equal(s_f, s_f_expected)


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
        self.assert_frame_equal(result, expected)


class TestVarious(BaseExtensionTests):
    @pytest.mark.xfail(reason="Don't know how to implement this correctly.")
    def test_concat_incompatible(self):
        s1 = pd.Series(["1 m"], dtype="unit")
        s2 = pd.Series(["1 ft"], dtype="unit")
        concatenated = pd.concat([s1, s2]).reset_index(drop=True)
        expected = pd.Series(["1 m", "0.3048 m"], dtype="unit")
        self.assert_series_equal(expected, concatenated)
        # :-( Returns converted to float.

    @pytest.mark.xfail(reason="Don't know how to implement this correctly.")
    def test_add_new_value_with_different_unit(self):
        s1 = pd.Series(["1 m"], dtype="unit")
        s1.at[1] = Quantity("1 ft")
        expected = pd.Series(["1.0 m", "0.3048 m"], dtype="unit")
        self.assert_series_equal(expected, s1)
        # :-( Returns converted to float.

    def test_set_value_with_different_unit(self):
        s1 = pd.Series(["1 m"], dtype="unit")
        s1[0] = Quantity("1 ft")
        expected = pd.Series(["0.3048 m"], dtype="unit")
        self.assert_series_equal(expected, s1)

    def test_unique(self):
        s = Series([1, np.nan, np.nan], dtype="unit[m]")
        unique = s.unique()
        expected = UnitsExtensionArray([1, np.nan], unit="m")
        assert unique.unit == expected.unit
        np.testing.assert_equal(expected.value, unique.value)
