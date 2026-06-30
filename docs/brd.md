# Business Requirements Document: Smart Cart Abandonment Predictor

## Problem Statement

E-commerce platforms lose 70% of shopping cart sessions to abandonment. Time-delayed email retargeting recovers < 5% of abandoned carts and degrades brand perception. A real-time, session-aware prediction system can intervene at the moment of highest intent — before the user leaves.

## High-Level Requirements (HLRs)

| ID | Requirement | Measure |
|----|-------------|---------|
| HLR-01 | Sub-200ms prediction latency | p99 API response time < 200ms under 100 concurrent requests |
| HLR-02 | High prediction accuracy | AUC-ROC ≥ 0.82 on held-out test set |
| HLR-03 | API availability | 99.9% uptime (< 8.7 hours downtime/year) |
| HLR-04 | False positive ceiling | < 10% of triggers on sessions that would have converted organically |
| HLR-05 | Stateless inference | No session state stored server-side in MVP; client sends full feature vector |
| HLR-06 | GDPR compliance | No PII (name, email, address) processed by the prediction API |

## User Personas

**Growth PM (Alex):** Owns checkout conversion rate. Wants a dashboard showing recovery rate, trigger volume, and incentive cost. Does not write code.

**Frontend Engineer (Sam):** Integrates `/predict` endpoint into checkout page JavaScript. Needs clear API contract, < 200ms latency, and a reliable availability SLA.

**Data Scientist (Priya):** Retrains the model monthly. Needs reproducible training pipeline, feature documentation, and evaluation metrics logged to stdout.

**Compliance Officer (Jordan):** Ensures GDPR compliance. Needs confirmation that no PII flows through the prediction API and that session IDs are pseudonymized.

## Out of Scope (MVP)

- Real-time model retraining (batch retraining on a schedule is sufficient)
- Server-side session state storage (client sends full feature vector per request)
- A/B testing of incentive types (single incentive ladder in MVP)
- Multi-tenant API (single-tenant deployment)
