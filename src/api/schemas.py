from pydantic import BaseModel, Field

from src.model.predict import Incentive


class SessionRequest(BaseModel):
    session_id: str
    cart_value: float = Field(ge=0.0)
    item_count: int = Field(ge=0)
    cart_adds: int = Field(ge=0)
    cart_removes: int = Field(ge=0)
    time_since_last_event_seconds: float = Field(ge=0.0)
    session_duration_seconds: float = Field(ge=0.0)
    page_views: int = Field(ge=1)
    checkout_start_attempts: int = Field(ge=0)
    device_type: str = "desktop"
    referrer_source: str = "direct"
    time_on_checkout_seconds: float = Field(ge=0.0, default=0.0)


class PredictionResponse(BaseModel):
    session_id: str
    risk_score: float
    incentive: Incentive
    high_risk: bool
