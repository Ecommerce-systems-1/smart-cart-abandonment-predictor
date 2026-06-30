# Ecommerce-systems Portfolio — Design Spec
**Date:** 2026-06-30
**Status:** Approved

---

## 1. Purpose & Goals

Build 23 GitHub projects specializing in Ecommerce, organized under the `Ecommerce-systems-1` GitHub Organization, in the following priority order:

1. **Portfolio (A)** — Showcase PM + engineering depth to ecommerce companies and consulting clients
2. **Learning (C)** — Systematically master different ecommerce domains through building
3. **Open-source (B)** — Reusable tools the community can deploy
4. **Commercial (D)** — Projects that eventually connect into a real, monetizable ecommerce platform

**Primary audience priority:** E-commerce companies (Shopify, Amazon, Instacart, DoorDash) > Consulting clients > Big Tech / FAANG > Startups

---

## 2. GitHub Organization Structure

**Org name:** `Ecommerce-systems-1`

**Org-level repos:**
- `.github` — Org profile README (`profile/README.md`): master table of all 23 projects with status badges, live demo links, tech stack icons, and visual architecture diagram
- `synthetic-data-toolkit` — Shared synthetic data generators (Faker-based, domain-specific models); published as a pip-installable package

---

## 3. Per-Project Standard Repository Structure

```
project-name/
├── README.md                  # PM_projects.md framework (why, who, metrics, architecture, tradeoffs, edge cases, what I'd do differently)
├── docs/
│   ├── 5-questions.md         # Customer, problem, solution, experience, success metrics
│   ├── brd.md                 # Business Requirements Document
│   ├── architecture.md        # System diagram + CAP/concurrency/idempotency decisions
│   └── data-model.md          # Schema, retention, archival
├── data/
│   └── generate.py            # Synthetic data generator (CLI: --scenario, --volume, --seed)
├── src/                       # Core implementation
├── tests/                     # Unit + integration tests with synthetic data fixtures
├── docker-compose.yml         # Local run in one command
├── .github/
│   └── workflows/             # CI: lint → test → deploy
└── deploy/                    # Free-tier deployment config
```

---

## 4. Deployment Strategy by Project Type

| Project Type | Platform | Rationale |
|---|---|---|
| ML / Data Science | HuggingFace Spaces (Gradio/Streamlit) | Free GPU, instant shareable URL |
| Backend APIs | Railway or Render | Free tier, Docker native, persistent |
| Frontend / Full-stack | Vercel | Free, CDN, Next.js native |
| Real-time / WebSocket | Railway | Supports long-running processes |
| Data dashboards | Streamlit Cloud | Free, Python native |

---

## 5. Tech Stack Philosophy

**Varied stacks** — each project uses the best tool for the job. No single stack enforced across all projects. This demonstrates adaptability and shows idiomatic use of each technology.

---

## 6. Synthetic Data Strategy

Every project includes a `data/generate.py` synthetic data generator with:
- **CLI flags:** `--scenario` (named scenario), `--volume` (record count), `--seed` (reproducibility)
- **Named scenarios:** happy path, edge cases, adversarial/stress cases per domain
- **Output formats:** CSV (batch), JSON stream (real-time), SQL seed file (database)
- **Realistic distributions:** Faker + domain-specific statistical models

**Four roles synthetic data plays per project:**
1. **Testing** — Reproducible datasets covering happy path, edge cases, failure modes
2. **Tradeoff validation** — Parameterized generators to stress-test architectural decisions
3. **Edge case coverage** — Adversarial generators (duplicates, negative inventory, ambiguous inputs)
4. **Live demo showcase** — "Demo mode" streaming pre-seeded data through the live system in real-time

---

## 7. Project Pipeline (23 Projects, Portfolio-Impact Order)

### Tier 1 — Lead Projects (Build First, Highest Impact)

| # | Project | Key Signal | Stack |
|---|---|---|---|
| 1 | Smart Cart Abandonment Predictor | ML + real-time scoring; $18B conversion problem | Python, FastAPI, XGBoost, Redis, Streamlit |
| 2 | Fraud Detection & Risk Scoring API | ML + behavioral signals + device fingerprinting; universal ecommerce need | Python, FastAPI, Scikit-learn, PostgreSQL, Railway |
| 3 | Semantic Search & Auto-Suggest Engine | NLP + vector embeddings; search = #1 conversion driver | Python, FastAPI, Sentence-Transformers, Qdrant, Next.js |
| 4 | AI Guardrails Service | LLM safety for internal + external chatbots; urgent 2026 problem | Python, FastAPI, Anthropic API, Guardrails AI, Redis, PostgreSQL |
| 5 | High-Volume Flash Sale Simulator | Distributed systems at 100x traffic; visually dramatic demo | Go, Redis, PostgreSQL, k6, Grafana |
| 6 | Dynamic Upsell & Cross-sell Engine | AOV optimization via recommendation ML | Python, FastAPI, Collaborative Filtering, React |

### Tier 2 — Business Value Projects (Consulting + Ecommerce Companies)

| # | Project | Key Signal | Stack |
|---|---|---|---|
| 7 | Recommendation System for Product Listing Ads | Two-tower ML for retail media networks; ad tech + ML intersection | Python, FastAPI, TensorFlow/PyTorch, Redis, PostgreSQL, React |
| 8 | Predictive Inventory Restock Planner | Supply chain forecasting dashboard; clear consulting ROI | Python, Prophet/ARIMA, Streamlit, PostgreSQL |
| 9 | Recommendation System for Market Basket Offers | Real-time bundle offers from cart co-occurrence patterns | Python, FastAPI, FP-Growth/MLxtend, PostgreSQL, Spark, React |
| 10 | Real-Time Competitor Price Monitor | Pricing intelligence; live price feed demo | Python, Scrapy, FastAPI, React, Redis |
| 11 | Subscription & Churn Mitigation Engine | Retention ML; LTV/churn metrics | Python, Scikit-learn, FastAPI, React, PostgreSQL |
| 12 | Customer Support Automation Hub | LLM triage; clear cost savings demo | Python, LangChain/Anthropic API, FastAPI, Next.js |
| 13 | Multi-Variant A/B/n Testing Framework | Native experimentation platform; statistical depth | Python, FastAPI, PostgreSQL, React, Docker |

### Tier 3 — Systems Depth Projects (FAANG Signal)

| # | Project | Key Signal | Stack |
|---|---|---|---|
| 14 | Distributed Order Management System | Split-shipping routing; CAP theorem trade-offs; backend depth | Go, PostgreSQL, Kafka, Docker |
| 15 | Dynamic Personalization & Feed Ranking | Feed ranking ML; affinity scoring pipeline | Python, FastAPI, Redis, React |
| 16 | Unified Customer Review Sentiment Analyzer | NLP + analytics; actionable insights from unstructured data | Python, HuggingFace Transformers, Streamlit |
| 17 | Affiliate & Influencer Attribution Service | First-party cookie attribution; data pipeline | Python, FastAPI, PostgreSQL, React |
| 18 | One-Click Localized Checkout System | Payments + geo routing + currency; UX + backend integration | Next.js, Node.js, Stripe API, PostgreSQL |

### Tier 4 — Specialized & Emerging Tech

| # | Project | Key Signal | Stack |
|---|---|---|---|
| 19 | Visual Search Product Matcher | CLIP-based computer vision; Pinterest/Google Lens signal | Python, CLIP, FastAPI, React, HuggingFace Spaces |
| 20 | Real-time Cold Chain Tracking System | IoT simulation + real-time alerting; niche but memorable | Python, MQTT, TimescaleDB, Grafana, React |
| 21 | Reverse Logistics & Returns Portal | Operations automation + return fraud detection | Next.js, Node.js, PostgreSQL, Railway |
| 22 | Automated SEO Taxonomy Generator | SEO + NLP + trend data; growth engineering | Python, FastAPI, Google Trends API, React |
| 23 | Global Catalog Localization Engine | i18n + compliance framework; international ecommerce | Node.js, PostgreSQL, React, Docker |

---

## 8. Per-Project Workflow (Phases 1–6)

### Phase 1: PM Documentation (PM_projects.md framework)
No implementation begins until all PM docs are complete and approved.

**1.1 — 5-Questions Document**
- Who is the customer?
- What is the problem and its size/opportunity?
- What is the high-level solution?
- What does the customer experience look like?
- What does success look like? What metrics are targeted?

**1.2 — Business Requirements Document (BRD)**
- Problem statement, High-Level Requirements (HLRs), User Personas

**1.3 — Functional & Non-Functional Requirements**
- Numbered FR list
- NFRs: SLA, p99 latency bounds, scalability, security standards (SOC2, GDPR where applicable)

**1.4 — Systems Architecture**
- CAP theorem alignment (CP vs. AP) with exact implications
- Concurrency & idempotency strategy (optimistic/pessimistic locking, idempotency keys)
- Failure modes + circuit breakers + DLQ + retry policies
- Architecture diagram (Mermaid in `docs/architecture.md`)

**1.5 — Data Model & Lifecycle**
- Schema, entity relationships, mutability
- Retention and archival policy

**1.6 — Phased Rollout**
- Phase 1 MVP scope
- Feature flagging strategy
- Zero-downtime migration plan

### Phase 2: Synthetic Data Design
- Define realistic entities and distributions for the domain
- Build `data/generate.py` with `--scenario`, `--volume`, `--seed` flags
- Define named scenarios: happy path, edge cases, adversarial/stress
- Outputs: CSV, JSON stream, SQL seed file

### Phase 3: Implementation
- Core feature in chosen stack
- Docker Compose for one-command local run
- CI pipeline: lint → test → deploy

### Phase 4: Testing
- Unit tests with synthetic data fixtures
- Integration tests against Docker services
- Edge case tests mapped from Phase 2 scenarios
- Load/stress tests where applicable (k6 for high-throughput projects)

### Phase 5: README & Documentation
Every README answers all 7 sections per `PM_projects.md`:
1. Why you built it (business problem + opportunity size)
2. Who it's for (personas)
3. Core metrics targeted
4. Tradeoffs made (CAP alignment, build vs. buy decisions)
5. Edge cases solved and why they matter
6. Architecture diagram (data flow: frontend → backend → DB → third-party APIs)
7. What you'd do differently (engineering maturity signal)

### Phase 6: Deploy & Publish
- Deploy to free-tier platform per deployment strategy table
- Add live demo badge + link to README
- Create GitHub repo under `Ecommerce-systems-1` org
- Pin top 6 repos on org profile
- Update org-level README with new project entry + status badge

---

## 9. Per-Project Checklist

```
[ ] PM docs complete (5Q + BRD + FRs/NFRs + Architecture + Data Model + Rollout)
[ ] Synthetic data generator built and tested (all scenarios pass)
[ ] Implementation complete
[ ] All tests passing (unit + integration + edge cases)
[ ] README complete (all 7 sections per PM_projects.md)
[ ] Docker Compose verified (one-command local run)
[ ] Live demo deployed and accessible
[ ] GitHub repo created under Ecommerce-systems org
[ ] Org-level README updated with new entry + status badge
```

---

## 10. Local Folder Structure

All 23 projects live under a single parent directory that mirrors the GitHub org:

```
VS Code Projects/
└── Ecommerce-systems/                              ← GitHub org mirror (open this in VS Code)
    ├── 01-smart-cart-abandonment-predictor/        ← Moved from VS Code Projects root
    ├── 02-fraud-detection-risk-scoring-api/
    ├── 03-semantic-search-auto-suggest-engine/
    ├── 04-ai-guardrails-service/
    ├── 05-high-volume-flash-sale-simulator/
    ├── 06-dynamic-upsell-crosssell-engine/
    ├── 07-product-listing-ads-recommender/
    ├── 08-predictive-inventory-restock-planner/
    ├── 09-market-basket-offers-recommender/
    ├── 10-real-time-competitor-price-monitor/
    ├── 11-subscription-churn-mitigation-engine/
    ├── 12-customer-support-automation-hub/
    ├── 13-multivariant-ab-testing-framework/
    ├── 14-distributed-order-management-system/
    ├── 15-dynamic-personalization-feed-ranking/
    ├── 16-customer-review-sentiment-analyzer/
    ├── 17-affiliate-influencer-attribution-service/
    ├── 18-one-click-localized-checkout-system/
    ├── 19-visual-search-product-matcher/
    ├── 20-real-time-cold-chain-tracking-system/
    ├── 21-reverse-logistics-returns-portal/
    ├── 22-automated-seo-taxonomy-generator/
    └── 23-global-catalog-localization-engine/
```

**Conventions:**
- Numbered prefixes (local only) keep projects in build order in VS Code Explorer
- GitHub repo names drop the number prefix (kebab-case only, e.g., `smart-cart-abandonment-predictor`)
- First action: move existing `Shopping Cart Abandonement Predictor/` into `Ecommerce-systems/01-smart-cart-abandonment-predictor/`

---

## 11. GitHub Repository Naming

```
Ecommerce-systems/                              ← GitHub Org
├── .github                                     ← Org profile README
├── synthetic-data-toolkit                      ← Shared pip-installable generators
├── smart-cart-abandonment-predictor
├── fraud-detection-risk-scoring-api
├── semantic-search-auto-suggest-engine
├── ai-guardrails-service
├── high-volume-flash-sale-simulator
└── ... (kebab-case, no numbers, 23 total)
```

---

## 13. Build Approach

**Sequential Deep-Dive (Approach A):** One project at a time, fully complete before moving to the next. Full PM_projects.md framework applied per project. Quality over speed — ecommerce company and consulting audiences read READMEs and architecture docs closely.

---

## 14. Glossary

| Term | Definition |
|---|---|
| CAP Theorem | A distributed system can guarantee only two of: Consistency, Availability, Partition Tolerance |
| CP System | Prioritizes Consistency + Partition Tolerance over Availability |
| AP System | Prioritizes Availability + Partition Tolerance over Consistency |
| Idempotency Key | A unique token ensuring duplicate API calls produce the same result without side effects |
| DLQ | Dead Letter Queue — stores messages that failed processing for later inspection/retry |
| p99 Latency | The 99th percentile response time — worst-case experienced by 1% of requests |
| AOV | Average Order Value — total revenue divided by number of orders |
| LTV | Lifetime Value — total revenue expected from a customer over their relationship |
| PLA | Product Listing Ad — sponsored product ad shown in search results |
| Two-Tower Model | ML architecture with separate encoder networks for users and items, joined at inference |
| FP-Growth | Frequent Pattern Growth — efficient algorithm for mining association rules in transaction data |
| Market Basket | Analysis of items frequently purchased together to generate bundle offers |
| Guardrails | Safety layer wrapping an LLM to enforce content policies, topic restrictions, and PII redaction |
| HLR | High-Level Requirement — a business need stated in measurable terms |
| NFR | Non-Functional Requirement — system quality attributes (latency, availability, scalability) |
| SLA | Service Level Agreement — committed uptime/performance threshold (e.g., 99.9% availability) |
| Blast Radius | The scope of impact if a component fails — smaller is better |
| Noisy Neighbor | A tenant consuming disproportionate shared resources, degrading others' performance |
| Write-Ahead Log | Database mechanism logging changes before applying them, enabling crash recovery |
