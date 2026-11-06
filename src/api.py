"""FastAPI inference service (Part B).

    GET  /health      liveness/readiness: is the model loaded? is MLflow reachable (if configured)?
    GET  /model-info  version, checksum, metrics and input schema of the loaded model
    POST /predict     price for one house (online inference)

Run:  uvicorn src.api:app --reload --host 127.0.0.1 --port 8000   ->  http://127.0.0.1:8000/docs

Configuration (environment variables, no hard-coded paths):
    MODEL_PATH            model file (default models/model.joblib; metadata must sit next to it)
    MLFLOW_TRACKING_URI   optional; when it is an http(s) URL, /health reports whether it is reachable
    LOG_LEVEL             debug | info | warning (default info)

Errors never expose internals: validation problems return 422 with field-level messages,
an unavailable model returns 503, anything unexpected returns 500 with a request id that
matches the server log line.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import __version__, schema
from src.predict import LoadedModel, ModelLoadError, load_model, predict, resolve_model_path

logger = logging.getLogger("house_api")
R = schema.VALID_RANGES


class HouseFeatures(BaseModel):
    """One house. Strict types: numbers must be JSON numbers, not strings."""

    model_config = ConfigDict(extra="forbid", strict=True,
                              json_schema_extra={"examples": [schema.EXAMPLE_HOUSE]})

    area_sqft: float = Field(..., ge=R["area_sqft"][0], le=R["area_sqft"][1],
                             description=schema.FEATURE_DESCRIPTIONS["area_sqft"])
    bedrooms: int = Field(..., ge=R["bedrooms"][0], le=R["bedrooms"][1],
                          description=schema.FEATURE_DESCRIPTIONS["bedrooms"])
    bathrooms: int = Field(..., ge=R["bathrooms"][0], le=R["bathrooms"][1],
                           description=schema.FEATURE_DESCRIPTIONS["bathrooms"])
    year_built: int = Field(..., ge=R["year_built"][0], le=R["year_built"][1],
                            description=schema.FEATURE_DESCRIPTIONS["year_built"])
    distance_to_center_km: float = Field(..., ge=R["distance_to_center_km"][0], le=R["distance_to_center_km"][1],
                                         description=schema.FEATURE_DESCRIPTIONS["distance_to_center_km"])
    has_garage: bool = Field(..., description=schema.FEATURE_DESCRIPTIONS["has_garage"])
    locality: Literal[schema.LOCALITIES] = Field(..., description=schema.FEATURE_DESCRIPTIONS["locality"])

    @model_validator(mode="after")
    def check_business_rule(self):
        problem = schema.business_rule_violation(self.bedrooms, self.bathrooms)
        if problem:
            raise ValueError(problem)
        return self


class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted price in INR lakhs")
    unit: str = "INR lakhs"
    model_version: str
    request_id: str


class ErrorResponse(BaseModel):
    error: str
    message: str
    details: list[dict] | None = None
    request_id: str


def _error(status: int, error: str, message: str, request: Request, details=None) -> JSONResponse:
    body = ErrorResponse(error=error, message=message, details=details,
                         request_id=getattr(request.state, "request_id", "-")).model_dump(exclude_none=True)
    return JSONResponse(status_code=status, content=body)


def _mlflow_status() -> dict:
    """Check MLflow with a real API route (MLflow's /health skips its Host-header check)."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith(("http://", "https://")):
        return {"configured": False}
    url = uri.rstrip("/") + "/api/2.0/mlflow/experiments/search?max_results=1"
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return {"configured": True, "reachable": resp.status == 200, "uri": uri}
    except urllib.error.HTTPError as e:
        return {"configured": True, "reachable": False, "uri": uri, "http_status": e.code}
    except (urllib.error.URLError, TimeoutError, OSError):
        return {"configured": True, "reachable": False, "uri": uri}


def create_app(model_path: str | None = None) -> FastAPI:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "info").upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started = time.time()
        app.state.model = None
        path = resolve_model_path(model_path)
        try:
            app.state.model = load_model(path)
            logger.info("model loaded: version=%s path=%s", app.state.model.version, path)
        except ModelLoadError as e:
            # Keep serving so /health can report the problem (503) instead of crash-looping.
            logger.error("model NOT loaded: %s", e)
        yield

    app = FastAPI(
        title="House Price Prediction API",
        version=__version__,
        description="Online inference for the Lab 7-9 house-price model. Prices are in INR lakhs.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_id_and_timing(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info("%s %s -> %s in %.1f ms (request_id=%s)", request.method, request.url.path,
                    response.status_code, (time.perf_counter() - start) * 1000, request.state.request_id)
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError):
        details = []
        for err in exc.errors():
            msg = err.get("msg", "invalid value").removeprefix("Value error, ")
            if err.get("type") == "json_invalid":       # loc holds a character position, not a field
                details.append({"field": "body", "message": f"{msg} at position {err['loc'][-1]}",
                                "type": "json_invalid"})
                continue
            loc = [str(p) for p in err.get("loc", []) if p != "body"]
            details.append({"field": ".".join(loc) or "body", "message": msg, "type": err.get("type")})
        return _error(422, "validation_error", "Request body failed validation", request, details)

    @app.exception_handler(Exception)
    async def on_unexpected_error(request: Request, exc: Exception):
        logger.exception("unhandled error (request_id=%s)", getattr(request.state, "request_id", "-"))
        return _error(500, "internal_error", "Prediction failed due to an internal error. "
                      "Quote the request_id when reporting it.", request)

    def _model(request: Request) -> LoadedModel | None:
        return request.app.state.model

    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "house-price-api", "docs": "/docs", "health": "/health"}

    @app.get("/health", summary="Service health and model readiness")
    def health(request: Request):
        model = _model(request)
        body = {"status": "ok" if model else "unavailable",
                "model_loaded": model is not None,
                "model_version": model.version if model else None,
                "uptime_s": round(time.time() - request.app.state.started, 1),
                "mlflow": _mlflow_status()}
        return JSONResponse(status_code=200 if model else 503, content=body)

    @app.get("/model-info", summary="Metadata of the loaded model")
    def model_info(request: Request):
        model = _model(request)
        if model is None:
            return _error(503, "model_unavailable", "Model is not loaded", request)
        m = model.metadata
        return {"model_version": model.version, "model_sha256": m["model_sha256"], "model_type": m["model_type"],
                "params": m["params"], "metrics": m["metrics"], "input_features": m["input_features"],
                "valid_ranges": m["valid_ranges"], "localities": m["localities"], "unit": m["unit"],
                "library_versions": m["library_versions"]}

    @app.post("/predict", response_model=PredictionResponse, summary="Predict the price of one house",
              responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse},
                         500: {"model": ErrorResponse}})
    def predict_one(payload: HouseFeatures, request: Request):
        model = _model(request)
        if model is None:
            return _error(503, "model_unavailable", "Model is not loaded; try again later", request)
        record = payload.model_dump()
        record["has_garage"] = int(record["has_garage"])
        value = float(predict(model, pd.DataFrame([record]))[0])
        logger.info("prediction %.4f lakhs (request_id=%s) input=%s", value, request.state.request_id,
                    json.dumps(record, sort_keys=True))
        return PredictionResponse(prediction=value, model_version=model.version,
                                  request_id=request.state.request_id)

    return app


app = create_app()
