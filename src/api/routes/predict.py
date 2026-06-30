import xgboost as xgb
from fastapi import APIRouter, Depends

from src.api.dependencies import get_model
from src.api.schemas import PredictionResponse, SessionRequest
from src.features.engineering import extract_features
from src.model.predict import predict

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
def predict_abandonment(
    request: SessionRequest,
    model: xgb.XGBClassifier = Depends(get_model),
) -> PredictionResponse:
    features = extract_features(request.model_dump())
    result = predict(model, features)
    return PredictionResponse(
        session_id=request.session_id,
        risk_score=result.risk_score,
        incentive=result.incentive,
        high_risk=result.risk_score >= 0.6,
    )
