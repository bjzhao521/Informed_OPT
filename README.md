# iOPT

Informed optimal path thresholds (iOPT) for compound hot, dry and windy days.

iOPT extends the optimal path threshold method (OPT; Zhao, Horvat & Gao, 2025) in three ways. The threshold grid is spaced evenly in rarity, r = -ln(1 - a), so that rare events are well resolved. Exactly N = round(p_t × n) days are selected, so every event definition has the same number of days. One threshold, here daily maximum temperature, can also be moved toward a physical value while N stays fixed.

## Files

| File | Content |
|---|---|
| `iopt.py` | The method (numpy only) |
| `run_example.py` | Runs iOPT for one county |
| `example_county.csv` | Daily gridMET data at Winnebago County, Illinois (FIPS 17201), 1980–2025 |
| `output/` | Results of `run_example.py` |

## Run

```bash
pip install numpy pandas
python run_example.py
```

Options:

```bash
python run_example.py --anchor 33.7          # soybean anchor instead of maize (34.8 C)
python run_example.py --rarity 0.005         # one rarity level only
python run_example.py --data my_county.csv   # columns: date (YYYYMMDD), tmax_C, rh_mean, wind_ms
```

## Event definitions

| Name | Variables (all upper tails) |
|---|---|
| H | daily maximum temperature |
| CDH | daily maximum temperature, relative-humidity deficit (100 − RH) |
| CHDW | daily maximum temperature, relative-humidity deficit, wind speed |

Thresholds are computed on growing-season (May–September) days. The target probability p_t refers to all days, so at p_t = 5‰ every definition selects 0.005 × 16,802 = 84 days in this county.

For CDH and CHDW, the temperature threshold has three options:

- `opt`: data-optimal
- `half`: moved halfway, in rarity, from the data-optimal value toward the anchor
- `anchored`: set at the anchor

With `half` and `anchored`, the other thresholds are re-optimized. If the anchor is below the growing-season median, or too high to give N days, it is set to the nearest feasible grid node and reported in `anchor_clipped`.

## Output

- `thresholds.csv`: one row per definition, threshold option and p_t. It gives the thresholds in physical units (`tmax_C`, `rh_below_pct`, `wind_ms`) and the number of event days.
- `events.csv`: the selected event days with their weather.

## Using `iopt()` directly

```python
from iopt import iopt

# T, RH, W: growing-season daily arrays; n_all: number of all days in the record
r = iopt([T, 100 - RH, W], p_t=0.005, n_total=n_all)                    # data-optimal
r = iopt([T, 100 - RH, W], p_t=0.005, n_total=n_all, anchor=34.8, f=1)  # anchored
r["mask"]   # True on the N event days
r["thr"]    # thresholds in the units of the inputs
```

## Data

The example data are from gridMET (Abatzoglou, 2013), extracted at the county centroid through Google Earth Engine (`IDAHO_EPSCOR/GRIDMET`). `tmax_C` is daily maximum temperature, `rh_mean` is daily mean relative humidity computed from Tmax, Tmin, RHmax and RHmin following FAO-56 (Allen et al., 1998), and `wind_ms` is daily mean 10-m wind speed. These are the inputs used in the paper.

## References

- Abatzoglou, J. T. (2013). Development of gridded surface meteorological data for ecological applications and modelling. *International Journal of Climatology*, 33, 121–131.
- Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration: Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56. FAO, Rome.
- Zhao, B., Horvat, C., & Gao, H. (2025). An optimal path threshold method for rigorously identifying extreme climate events. *Environmental Research Letters*, 20, 024048.

## License

MIT
