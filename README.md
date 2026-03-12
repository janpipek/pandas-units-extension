# Extension pandas dtype and array for physical units

This python package extends the popular data science library [pandas](https://github.com/pandas-dev/pandas/) with the physical units of the astrophysics library [astropy](https://www.astropy.org/). The package is compatible with modern pandas, although some features are only working with pandas version 3.0 or an upcoming pandas release.

## History

Originally created for PyCon CZ 2019 (and later BI Fórum Budapest / PyData Cambridge the same year) by Jan Pipek.
Updated to be compatible with modern pandas since 2025 by Julian Harbeck during the RACCOON project at Technische Universität Berlin.

## Installation

```bash
pip install pandas-units-extensions
```

For development:

```bash
pip install -e .
```

## Examples

```python
>>> import pandas as pd
>>> import pandas_units_extension as pue

>>> temps = pd.DataFrame({
        "city": ["Prague", "Kathmandu", "Catania", "Boston"],
        "temperature": pd.Series([20, 22, 31, 16], dtype="unit[deg_C]")
    })
>>> temps["temperature"].units.to("deg_F")
0    68.0 deg_F
1    71.6 deg_F
2    87.8 deg_F
3    60.8 deg_F
Name: temperature, dtype: unit[deg_F]


>>> df = pd.DataFrame({
        "distance": pd.Series([10, 12, 22, 18], dtype="unit[km]"),
        "time": pd.Series([50, 60, 120, 108], dtype="unit[min]")
    })
>>> speed = df["distance"] / df["time"]
>>> speed.units.to_si()
0     3.333333333333334 m / s
1     3.333333333333334 m / s
2    3.0555555555555554 m / s
3    2.7777777777777777 m / s
dtype: unit[m / s]
```

See [doc/units.ipynb](doc/units.ipynb) for more.

## Links

- <https://www.astropy.org/>
- <https://pint.readthedocs.io/en/0.10.1/pint-pandas.html> - Another library supporting units inside pandas.
