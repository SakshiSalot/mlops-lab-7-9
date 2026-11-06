# Local vs docker container predictions (Part D)

Run 2026-11-06 10:55:58 UTC with `python scripts/compare_predictions.py --url http://localhost:8000`. Tolerance: absolute difference <= 1e-06 lakhs.

| # | Input | Local prediction | API prediction | Abs diff | Within tolerance |
|---|---|---|---|---|---|
| 1 | 1850 sqft, 3bd/2ba, 2012, 6.5 km, North | 104.4156157929 | 104.4156157929 | 0.00e+00 | yes |
| 2 | 650 sqft, 1bd/1ba, 1985, 22.0 km, East | 28.7331884810 | 28.7331884810 | 0.00e+00 | yes |
| 3 | 1200 sqft, 2bd/2ba, 2005, 12.3 km, West | 59.0905049526 | 59.0905049526 | 0.00e+00 | yes |
| 4 | 2400 sqft, 4bd/3ba, 2019, 3.1 km, Central | 157.7708124999 | 157.7708124999 | 0.00e+00 | yes |
| 5 | 3100 sqft, 5bd/4ba, 1998, 8.75 km, South | 148.1507608269 | 148.1507608269 | 0.00e+00 | yes |
| 6 | 3900.5 sqft, 6bd/5ba, 2024, 0.8 km, North | 222.8399702578 | 222.8399702578 | 0.00e+00 | yes |

**Max absolute difference:** 0.00e+00 -> **consistent**

Same model artifact: yes (local SHA-256 `b09c57dd3b346698...`, API reports `b09c57dd3b346698...`).

Libraries the model was trained with (from model_metadata.json): python 3.11.17, scikit-learn 1.9.1, numpy 2.4.6, pandas 3.0.6, joblib 1.6.0.
