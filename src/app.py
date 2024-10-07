from fastapi import FastAPI, HTTPException, Response
from models import ComplaintData, TopicPrediction, PredictionResponse
from typing import List

from config import Config
from inference import InferenceModel
from utils import to_snake_case

import uvicorn

from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

import traceback

config = Config()

registry = CollectorRegistry()
counters_dict = {}
for key, value in config.topic_mapping.items():
    counters_dict[key] = Counter(
        name=to_snake_case(value),
        documentation=f'Count of {value} topic complaints',
        registry=registry
    )

class PrometheusMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, registry):
        super().__init__(app)
        self.registry = registry

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        return response


app = FastAPI()
predictor = InferenceModel()

@app.post("/api/predict")
async def predict(complaints: List[ComplaintData]):
    try:
        print(complaints)
        complaints_texts = [{'complaint_what_happened': complaint.source.complaint_what_happened} for complaint in complaints]
        predictions = predictor.predict(complaints_texts)
        response = []
        for i, row in enumerate(predictions):
            print(row)
            complaint = complaints[i]
            counters_dict[row.prediction].inc()
            response.append(TopicPrediction(complaint_id=complaint.id, prediction=row["prediction"], complaint_text=row["complaint_what_happened"]))
        return PredictionResponse(data=response)
    except Exception as e:
        _traceback = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}, {_traceback}")

@app.get("/api/health")
async def health():
    return {"status": "healthy"}

@app.get("/api/ready")
async def ready():
    # Add any readiness check logic here
    return {"status": "ready"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


app.add_middleware(PrometheusMiddleware, registry=registry)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.backend_host,
        port=config.backend_port,
    )