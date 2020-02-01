# Extension pandas dtype and array for physical units

**Warning:** Not fully compatible with pandas yet, especially not so for 1.0. Some operations may or may not work, while common ones should.

## History

Originally created for PyCon CZ 2019 (and later BI Fórum Budapest / PyData Cambridge the same year).

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
import pandas as pd
import pandas_units_extension as _

temps = pd.DataFrame({
    "city": ["Prague", "Kathmandu", "Catania", "Boston"],
    "temperature": pd.Series([20, 22, 31, 16], dtype="unit[deg_C]")
})
temps["temperature"].units.to("deg_F")

...

df = pd.DataFrame({
    "distance": pd.Series([10, 12, 22, 18], dtype="unit[km]"),
    "time": pd.Series([50, 60, 120, 108], dtype="unit[min]")
})
speed = df["distance"] / df["time"]
speed.units.to_si()
```

See [doc/units.ipynb](doc/units.ipynb) for more.

## Links

- <https://www.astropy.org/>
- <https://pint.readthedocs.io/en/0.10.1/pint-pandas.html> - Another library supporting units inside pandas.
