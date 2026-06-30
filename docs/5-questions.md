# 5-Questions: Smart Cart Abandonment Predictor

## 1. Who is the customer?

**Primary:** Growth Engineers and Product Managers at mid-to-large e-commerce companies (50K+ monthly orders) who own checkout conversion rate as a core KPI.

**Secondary:** Frontend Engineers integrating the prediction API into checkout flows; Data Scientists responsible for model monitoring and retraining.

## 2. What is the customer problem statement?

Approximately 70% of online shopping carts are abandoned, representing $18B in lost US e-commerce revenue annually (Baymard Institute, 2024). Current solutions rely on **post-abandonment email retargeting** — triggered hours after the user has left. These are too slow (user intent is cold), too blunt (same message for all users), and too expensive (email deliverability costs, discount fatigue).

**Problem size:** A retailer doing $10M/month in GMV with a 70% abandonment rate has ~$23M in abandoned cart value monthly. Recovering 15% of at-risk carts adds ~$3.5M/month.

## 3. What is the high-level solution?

A real-time ML scoring engine embedded in the checkout flow that:
1. Receives session event signals (cart value, page dwell time, remove actions, device type) as they happen
2. Returns an abandonment risk score (0.0–1.0) within 200ms
3. Recommends the minimum-cost incentive to retain the session (urgency message → free shipping → 10% discount)
4. Triggers the incentive on the frontend before the user navigates away

## 4. What does the customer experience look like?

- User adds items to cart on a product page
- System silently scores every session event with no perceptible latency
- At risk score ≥ 0.6, the frontend displays a contextual banner or modal with the recommended incentive
- User either converts (success) or exits (session logged for model improvement)
- Conversion and trigger events feed back into the next model retraining cycle

## 5. What does success look like?

| Metric | Target | Measurement |
|--------|--------|-------------|
| Cart recovery rate | 15–25% of scored at-risk sessions | Conversions after incentive trigger / total incentive triggers |
| p99 prediction latency | < 200ms end-to-end | API response time percentile |
| False positive rate | < 10% | Incentive triggers where user would have converted anyway |
| Model AUC-ROC | ≥ 0.82 | Held-out test set evaluation |
| API availability | 99.9% | Uptime monitoring |
| AOV of recovered carts | ≥ baseline AOV ($85) | Average order value of triggered + converted sessions |
