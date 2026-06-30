from fastapi import FastAPI

from src.api.routes import health, predict

app = FastAPI(
    title="Smart Cart Abandonment Predictor",
    description="Real-time ML engine scoring cart abandonment risk and recommending incentives.",
    version="1.0.0",
)

app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, tags=["Prediction"])
