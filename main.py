from typing import Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel
from joblib import load
import uvicorn
from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter, Histogram

# Prometheus Metrics
prediction_counter = Counter('num_prediction_requests', "Number of 'prediction' requests made")
model_information_counter = Counter('num_model_info_requests', "Number of 'model_information' requests made")
prediction_hist = Histogram('prediction_output_distribution', 'Distribution of prediction outputs')
prediction_score_hist = Histogram('prediction_score_distribution', 'Distribution of prediction scores')
prediction_latency_hist = Histogram('prediction_latency_distribution', 'Distribution of prediction response latency')

app = FastAPI()
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

model = load("anomaly-model.joblib")

class PredictionRequest(BaseModel):
    feature_vector: List[float]
    score: Optional[bool] = False

@app.post("/prediction")
def predict(req: PredictionRequest) -> Dict:
    start = datetime.now()
    prediction_counter.inc()

    response = {}
    prediction = model.predict([req.feature_vector])
    response["is_inlier"] = int(prediction[0])
    prediction_hist.observe(response["is_inlier"])

    if req.score:
        score = model.score_samples([req.feature_vector])
        response["anomaly_score"] = score[0]
        prediction_score_hist.observe(response["anomaly_score"])

    latency = datetime.now() - start
    prediction_latency_hist.observe(latency.total_seconds())
    return response

@app.get("/model_information")
def model_information():
    model_information_counter.inc()
    return model.get_params()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
