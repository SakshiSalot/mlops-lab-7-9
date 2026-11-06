# API validation (B5): local uvicorn

Target `http://127.0.0.1:8000`, run 2026-11-06 10:41:07 UTC with `python scripts/api_check.py`.

| Test | Expected behaviour | Observed status/result | Pass |
|---|---|---|---|
| Valid input | Prediction returned (200) | 200: prediction 104.4156 INR lakhs (model 1.0.0+b09c57dd3b34) | PASS |
| Missing field (bedrooms) | Validation error (422) | 422: bedrooms - Field required | PASS |
| Wrong type (area_sqft as text) | Validation error (422) | 422: area_sqft - Input should be a valid number | PASS |
| Invalid business value (1 bedroom, 4 bathrooms) | Controlled error (422) | 422: body - bathrooms (4) cannot exceed bedrooms + 1 (2) | PASS |
| Out-of-range value (year_built 2031) | Validation error (422) | 422: year_built - Input should be less than or equal to 2026 | PASS |
| Malformed JSON body | Validation error (422) | 422: body - JSON decode error at position 19 | PASS |
| Health check | Service reports healthy (200) | 200: status=ok, model_loaded=True | PASS |
| Valid input again after the errors | Still 200: service did not crash | 200: prediction 104.4156 INR lakhs (model 1.0.0+b09c57dd3b34) | PASS |

## Example requests and responses

**Valid input** - `POST /predict` -> `200`

Request:
```json
{
  "area_sqft": 1850,
  "bedrooms": 3,
  "bathrooms": 2,
  "year_built": 2012,
  "distance_to_center_km": 6.5,
  "has_garage": true,
  "locality": "North"
}
```
Response:
```json
{
  "prediction": 104.41561579290465,
  "unit": "INR lakhs",
  "model_version": "1.0.0+b09c57dd3b34",
  "request_id": "35eb9e819a49"
}
```

**Missing field (bedrooms)** - `POST /predict` -> `422`

Request:
```json
{
  "area_sqft": 1850,
  "bathrooms": 2,
  "year_built": 2012,
  "distance_to_center_km": 6.5,
  "has_garage": true,
  "locality": "North"
}
```
Response:
```json
{
  "error": "validation_error",
  "message": "Request body failed validation",
  "details": [
    {
      "field": "bedrooms",
      "message": "Field required",
      "type": "missing"
    }
  ],
  "request_id": "7744f3c45d7e"
}
```

**Invalid business value (1 bedroom, 4 bathrooms)** - `POST /predict` -> `422`

Request:
```json
{
  "area_sqft": 1850,
  "bedrooms": 1,
  "bathrooms": 4,
  "year_built": 2012,
  "distance_to_center_km": 6.5,
  "has_garage": true,
  "locality": "North"
}
```
Response:
```json
{
  "error": "validation_error",
  "message": "Request body failed validation",
  "details": [
    {
      "field": "body",
      "message": "bathrooms (4) cannot exceed bedrooms + 1 (2)",
      "type": "value_error"
    }
  ],
  "request_id": "b05f12539708"
}
```

