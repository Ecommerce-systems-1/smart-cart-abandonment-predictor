from dataclasses import dataclass
from enum import Enum

import xgboost as xgb

from src.features.engineering import SessionFeatures, features_to_array


class Incentive(str, Enum):
    NONE = "none"
    URGENCY = "urgency_message"
    FREE_SHIPPING = "free_shipping"
    DISCOUNT_10 = "discount_10_percent"


@dataclass
class Prediction:
    risk_score: float
    incentive: Incentive


def recommend_incentive(risk_score: float) -> Incentive:
    if risk_score < 0.3:
        return Incentive.NONE
    elif risk_score < 0.6:
        return Incentive.URGENCY
    elif risk_score < 0.8:
        return Incentive.FREE_SHIPPING
    return Incentive.DISCOUNT_10


def predict(model: xgb.XGBClassifier, features: SessionFeatures) -> Prediction:
    x = features_to_array(features).reshape(1, -1)
    risk_score = float(model.predict_proba(x)[0][1])
    return Prediction(risk_score=risk_score, incentive=recommend_incentive(risk_score))
