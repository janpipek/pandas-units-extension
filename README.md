# Extension pandas dtype and array for physical units

## Installation

```bash
pip install -e .   # Will be in PyPI maybe
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
