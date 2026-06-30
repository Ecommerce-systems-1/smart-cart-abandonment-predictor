# Smart Cart Abandonment Predictor

> Real-time ML engine scoring cart abandonment risk and triggering the minimum-cost recovery incentive before the user leaves.

[![CI](https://github.com/Ecommerce-systems-1/smart-cart-abandonment-predictor/actions/workflows/ci.yml/badge.svg)](https://github.com/Ecommerce-systems-1/smart-cart-abandonment-predictor/actions)
[![Live Demo](https://img.shields.io/badge/HuggingFace-Live%20Demo-blue)](https://huggingface.co/spaces/RishabhHajela/smart-cart-abandonment-predictor)
[![Part of Ecommerce-systems](https://img.shields.io/badge/Ecommerce--systems-Project%2001%2F23-orange)](https://github.com/Ecommerce-systems-1)

## Why I Built This

~70% of online shopping carts are abandoned, costing US e-commerce ~$18B/year (Baymard Institute). Current retargeting solutions fire hours after abandonment — by then, intent is cold. This project proves you can score abandonment risk **in real-time** (< 200ms) and intervene at the moment of highest intent with a minimum-cost incentive ladder.

## Who It's For

- **Growth PMs** at e-commerce companies who own checkout conversion rate
- **Frontend Engineers** integrating a risk signal into checkout flows
- **Data Scientists** building cart recovery ML pipelines

## Core Metrics Targeted

| Metric | Target |
|--------|--------|
| p99 prediction latency | < 200ms |
| Model AUC-ROC | >= 0.82 |
| Cart recovery rate | 15-25% of at-risk sessions |
| False positive rate | < 10% |

## Architecture

```
Browser → POST /predict (FastAPI) → Feature Engineering → XGBoost Model → Incentive Engine → Response
                                         ↑
                               data/generate.py (synthetic training data)
```

**CAP alignment:** AP — availability over consistency. A slightly stale model (trained yesterday) is better than a prediction service that returns errors. Inference is stateless and thread-safe.

**Incentive ladder:**

| Risk Score | Action |
|-----------|--------|
| < 0.30 | No intervention |
| 0.30 - 0.59 | Urgency message ("Only 2 left!") |
| 0.60 - 0.79 | Free shipping offer |
| >= 0.80 | 10% discount code |

## Edge Cases Solved

- **Zero cart value:** Feature extractor clamps to 0.0 — no division errors; model returns low risk
- **Cart removes > adds:** `cart_remove_ratio` is safe via `(removes / (adds + 1))` denominator guard
- **Zero page_views:** Pydantic validator rejects `page_views < 1` at the API boundary (422 response)
- **Tablet device:** Mapped to `device_type_mobile = 0` — insufficient training signal to separate from desktop; documented in tradeoffs

## Tradeoffs

**XGBoost over deep learning:** 50K training samples is too small for a neural net to outperform gradient boosting. XGBoost trains in < 30s, loads in < 100ms, and runs inference in < 2ms. Revisit with neural collaborative filtering at 10M+ sessions.

**Stateless inference:** The client sends the full feature vector per request. No server-side session store in MVP. This simplifies scaling but puts session aggregation burden on the client.

**Synthetic training data only:** No real customer data used or required. Model may underperform on real data — validate against real sessions before production use.

**Tablet mapped to desktop:** Insufficient signal in synthetic data. Real deployment should use three-way encoding once real session data is available.

## What I'd Do Differently

1. **Feature store with Redis:** Cache per-session feature state server-side to avoid clients computing session-level aggregates.
2. **Online learning:** Retrain on a rolling 30-day window of labeled sessions rather than synthetic data.
3. **Calibration layer:** Add Platt scaling or isotonic regression to make `risk_score` a true probability.
4. **A/B testing integration:** Wire the incentive ladder to an experimentation platform to test urgency vs. free shipping vs. discount effectiveness.

## Quick Start

```bash
git clone https://github.com/Ecommerce-systems-1/smart-cart-abandonment-predictor
cd smart-cart-abandonment-predictor
docker compose up --build
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

Or train and run directly:

```bash
pip install -r requirements-dev.txt
python -m src.model.train          # Train model (~30s)
uvicorn src.api.main:app --reload  # Start API
streamlit run demo/app.py          # Start live demo
```

## Generate Synthetic Data

```bash
python data/generate.py --scenario high_risk --volume 5000 --seed 42 > data/high_risk.csv
python data/generate.py --scenario happy_path --volume 5000 --seed 42 > data/happy_path.csv
python data/generate.py --scenario edge_cases --volume 1000 --seed 42 > data/edge_cases.csv
```

## Run Tests

```bash
pytest tests/ -v
# 35 tests: 12 feature engineering + 13 model training/inference + 10 API
```

## Project Structure

```
src/features/   Feature extraction from session event dict
src/model/      XGBoost training pipeline + inference + incentive logic
src/api/        FastAPI app, Pydantic schemas, prediction route
demo/           Streamlit live demo (streams synthetic sessions in real-time)
data/           Synthetic data generator (4 scenarios, CLI-configurable)
tests/          35 unit + integration tests
```
