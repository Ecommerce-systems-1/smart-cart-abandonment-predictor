# Ecommerce-systems Portfolio Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the `Ecommerce-systems` local folder structure, move project 1 into it, create the GitHub org, and scaffold the two shared org-level repos (org profile README and synthetic-data-toolkit).

**Architecture:** All 23 projects live under `VS Code Projects\Ecommerce-systems\` mirroring the GitHub org. Each project is a numbered subdirectory (`01-...` through `23-...`) and an independent git repo. Two shared repos live at org level: `.github` (org profile README) and `synthetic-data-toolkit` (shared generators pip package).

**Tech Stack:** PowerShell, Git, GitHub CLI (`gh`)

---

### Task 1: Create Local Ecommerce-systems Directory Structure

**Files:**
- Create: `VS Code Projects\Ecommerce-systems\`
- Create: `VS Code Projects\Ecommerce-systems\01-smart-cart-abandonment-predictor\`

- [ ] **Step 1: Create parent directory**

Run in PowerShell:
```powershell
New-Item -ItemType Directory -Path "C:\Users\rishh\OneDrive\Documents\Claude\Projects\VS Code Projects\Ecommerce-systems"
```
Expected: `Directory: ...VS Code Projects` with `Ecommerce-systems` listed

- [ ] **Step 2: Create project 1 subdirectory**

Run in PowerShell:
```powershell
New-Item -ItemType Directory -Path "C:\Users\rishh\OneDrive\Documents\Claude\Projects\VS Code Projects\Ecommerce-systems\01-smart-cart-abandonment-predictor"
```
Expected: Directory created with no errors

---

### Task 2: Move Existing Project 1 Files Into New Structure

**Files:**
- Move: `Shopping Cart Abandonement Predictor\PM_projects.md` → `Ecommerce-systems\01-smart-cart-abandonment-predictor\PM_projects.md`
- Move: `Shopping Cart Abandonement Predictor\docs\` → `Ecommerce-systems\01-smart-cart-abandonment-predictor\docs\`

- [ ] **Step 1: Move PM_projects.md**

Run in PowerShell:
```powershell
$src = "C:\Users\rishh\OneDrive\Documents\Claude\Projects\VS Code Projects\Shopping Cart Abandonement Predictor"
$dst = "C:\Users\rishh\OneDrive\Documents\Claude\Projects\VS Code Projects\Ecommerce-systems\01-smart-cart-abandonment-predictor"
Move-Item -Path "$src\PM_projects.md" -Destination "$dst\PM_projects.md"
```
Expected: No errors

- [ ] **Step 2: Move docs directory**

Run in PowerShell:
```powershell
Move-Item -Path "$src\docs" -Destination "$dst\docs"
```
Expected: `docs\superpowers\specs\` and `docs\superpowers\plans\` moved successfully

- [ ] **Step 3: Verify new structure**

Run in PowerShell:
```powershell
Get-ChildItem -Recurse "$dst"
```
Expected output includes:
```
docs\superpowers\specs\2026-06-30-ecommerce-portfolio-design.md
docs\superpowers\plans\2026-06-30-portfolio-bootstrap.md
docs\superpowers\plans\2026-06-30-project-01-smart-cart-abandonment-predictor.md
PM_projects.md
```

- [ ] **Step 4: Remove old empty directory**

Confirm the old directory is now empty, then remove it:
```powershell
Get-ChildItem "$src"   # must show 0 items
Remove-Item -Path "$src" -Recurse -Force
```
Expected: Old `Shopping Cart Abandonement Predictor\` directory removed

---

### Task 3: Initialize Git for Project 1

**Files:**
- Create: `01-smart-cart-abandonment-predictor\.gitignore`

- [ ] **Step 1: Init git repo**

Run in PowerShell:
```powershell
Set-Location "C:\Users\rishh\OneDrive\Documents\Claude\Projects\VS Code Projects\Ecommerce-systems\01-smart-cart-abandonment-predictor"
git init
git branch -M main
```
Expected: `Initialized empty Git repository in ...01-smart-cart-abandonment-predictor/.git/`

- [ ] **Step 2: Create .gitignore**

Create file `C:\Users\rishh\OneDrive\Documents\Claude\Projects\VS Code Projects\Ecommerce-systems\01-smart-cart-abandonment-predictor\.gitignore`:
```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
.env

# Models (generated artifacts — train locally)
models/*.json
models/*.pkl
!models/.gitkeep

# Data outputs
data/output/

# Test artifacts
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Initial commit**

Run in PowerShell:
```powershell
git add PM_projects.md docs/ .gitignore
git commit -m "chore: bootstrap project 1 — move files into Ecommerce-systems structure"
```
Expected: Commit created on `main`

---

### Task 4: Create GitHub Organization

> **Manual browser step — cannot be automated via CLI.**

- [ ] **Step 1: Create the org**

1. Navigate to: `https://github.com/organizations/plan`
2. Organization name: `Ecommerce-systems`
3. Plan: Free
4. Complete the setup wizard
5. Click "Create organization"

Expected: GitHub org available at `https://github.com/Ecommerce-systems`

---

### Task 5: Create Org Profile README Repo

**Files:**
- Create locally: `Ecommerce-systems\.github\profile\README.md`
- GitHub repo: `Ecommerce-systems/.github`

- [ ] **Step 1: Create local .github directory and init git**

Run in PowerShell:
```powershell
$orgDir = "C:\Users\rishh\OneDrive\Documents\Claude\Projects\VS Code Projects\Ecommerce-systems\.github"
New-Item -ItemType Directory -Path "$orgDir\profile"
Set-Location $orgDir
git init
git branch -M main
```

- [ ] **Step 2: Create org README**

Create file `profile\README.md`:
```markdown
# Ecommerce-systems

23 production-grade ecommerce engineering projects — from ML-powered cart recovery to distributed order management at scale.

Each project follows a rigorous PM → architecture → synthetic data → TDD implementation → live demo pipeline.

## Projects

| # | Project | Domain | Stack | Status |
|---|---------|--------|-------|--------|
| 01 | [Smart Cart Abandonment Predictor](https://github.com/Ecommerce-systems/smart-cart-abandonment-predictor) | Checkout & Conversion | Python · XGBoost · FastAPI · Streamlit | 🟢 Live |
| 02 | Fraud Detection & Risk Scoring API | Checkout & Conversion | Python · FastAPI · Scikit-learn | 🔜 Coming soon |
| 03 | Semantic Search & Auto-Suggest Engine | Search & Discovery | Python · Sentence-Transformers · Qdrant · Next.js | 🔜 Coming soon |
| 04 | AI Guardrails Service | Platform Trust | Python · Anthropic API · FastAPI | 🔜 Coming soon |
| 05 | High-Volume Flash Sale Simulator | Platform Infrastructure | Go · Redis · PostgreSQL · k6 | 🔜 Coming soon |
| 06–23 | Full pipeline in progress | Various | Varied | 🔜 Coming soon |

## Shared Tooling

[`synthetic-data-toolkit`](https://github.com/Ecommerce-systems/synthetic-data-toolkit) — Domain-specific synthetic data generators used across all 23 projects. Pip-installable.

## Design Philosophy

Every project in this org is designed around the same framework:
- **Problem-first:** 5-Questions doc + BRD before a line of code is written
- **Systems thinking:** Explicit CAP theorem trade-offs, concurrency strategy, failure modes
- **Synthetic data:** Parameterized generators for happy path, edge cases, and stress tests
- **Live demos:** Every project is deployed and accessible without local setup
```

- [ ] **Step 3: Commit and push**

Run in PowerShell:
```powershell
git add profile\README.md
git commit -m "chore: initialize org profile README"
gh repo create Ecommerce-systems/.github --public --source=. --remote=origin --push
```
Expected: Repo visible at `https://github.com/Ecommerce-systems/.github`; org profile README renders at `https://github.com/Ecommerce-systems`

---

### Task 6: Scaffold synthetic-data-toolkit Repo

**Files:**
- Create: `Ecommerce-systems\synthetic-data-toolkit\src\ecommerce_synthetic\__init__.py`
- Create: `Ecommerce-systems\synthetic-data-toolkit\src\ecommerce_synthetic\base.py`
- Create: `Ecommerce-systems\synthetic-data-toolkit\pyproject.toml`
- Create: `Ecommerce-systems\synthetic-data-toolkit\README.md`

- [ ] **Step 1: Create directory structure and init git**

Run in PowerShell:
```powershell
$tkDir = "C:\Users\rishh\OneDrive\Documents\Claude\Projects\VS Code Projects\Ecommerce-systems\synthetic-data-toolkit"
New-Item -ItemType Directory -Path "$tkDir\src\ecommerce_synthetic"
Set-Location $tkDir
git init
git branch -M main
```

- [ ] **Step 2: Create package files**

Create `src\ecommerce_synthetic\__init__.py`:
```python
"""Ecommerce-systems synthetic data toolkit."""
__version__ = "0.1.0"
```

Create `src\ecommerce_synthetic\base.py`:
```python
import random
import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible synthetic data generation."""
    random.seed(seed)
    np.random.seed(seed)
```

Create `pyproject.toml`:
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ecommerce-synthetic"
version = "0.1.0"
description = "Synthetic data generators for Ecommerce-systems projects"
requires-python = ">=3.11"
dependencies = [
    "faker>=25.0",
    "numpy>=1.26",
    "pandas>=2.0",
]

[project.urls]
Repository = "https://github.com/Ecommerce-systems/synthetic-data-toolkit"
```

Create `README.md`:
```markdown
# synthetic-data-toolkit

Shared synthetic data generators for [Ecommerce-systems](https://github.com/Ecommerce-systems) projects.

## Install

```bash
pip install git+https://github.com/Ecommerce-systems/synthetic-data-toolkit.git
```

## Usage

Individual project `data/generate.py` scripts import domain-specific generators from this package as patterns are extracted and generalized.
```

Create `.gitignore`:
```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.venv/
dist/
build/
```

- [ ] **Step 3: Commit and push**

Run in PowerShell:
```powershell
git add .
git commit -m "chore: scaffold synthetic-data-toolkit package"
gh repo create Ecommerce-systems/synthetic-data-toolkit --public --source=. --remote=origin --push --description "Shared synthetic data generators for Ecommerce-systems projects"
```
Expected: Repo at `https://github.com/Ecommerce-systems/synthetic-data-toolkit`

---

### Task 7: Push Project 1 to GitHub

- [ ] **Step 1: Create GitHub repo and push**

Run in PowerShell:
```powershell
Set-Location "C:\Users\rishh\OneDrive\Documents\Claude\Projects\VS Code Projects\Ecommerce-systems\01-smart-cart-abandonment-predictor"
gh repo create Ecommerce-systems/smart-cart-abandonment-predictor --public --source=. --remote=origin --push --description "ML engine predicting cart abandonment risk in real-time to trigger dynamic incentives"
```
Expected: Repo at `https://github.com/Ecommerce-systems/smart-cart-abandonment-predictor`

- [ ] **Step 2: Pin top repos on org profile**

1. Open `https://github.com/Ecommerce-systems`
2. Click "Customize your organization"
3. Pin: `smart-cart-abandonment-predictor`, `synthetic-data-toolkit`
4. Save

- [ ] **Step 3: Verify final org structure**

Open in browser: `https://github.com/Ecommerce-systems`

Expected: 3 repos visible — `.github`, `synthetic-data-toolkit`, `smart-cart-abandonment-predictor`. Org profile README renders correctly with project table.

**Bootstrap complete. Proceed to:** `docs/superpowers/plans/2026-06-30-project-01-smart-cart-abandonment-predictor.md`
