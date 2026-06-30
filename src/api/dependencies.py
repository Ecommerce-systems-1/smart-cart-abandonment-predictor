from functools import lru_cache
from pathlib import Path

import xgboost as xgb

MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "cart_abandonment_model.json"


@lru_cache(maxsize=1)
def get_model() -> xgb.XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python -m src.model.train"
        )
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    return model
