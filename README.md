# Grade Change Intelligence in Paper Making Process

An AI decision-support system that predicts, prevents, and explains quality deviations during paper-machine **grade changes** — reducing off-spec waste and stabilization time.

> **Honeywell Hackathon — Problem Statement 5**
> Shivam Sukhija · 20474401

**🔗 Live Demo (video + files):** https://drive.google.com/drive/folders/1oMQerlZm7KJ05tllAi2ms3LugmgCb49f?usp=drive_link
**💻 Repository:** https://github.com/sukhijashivam/Grade_Change_Intelligence

---

## The Problem

Paper machines produce different **grades** (paper weights). When switching grade (e.g. 100 → 120 GSM), the main quality variable **Basis Weight** swings off-target and takes 40–70 minutes to stabilize. During this window the mill produces **off-spec paper (waste)**. Existing control (Honeywell QCS / MD-MPC) *executes* the change but does **not learn** from history.

**Goal:** an intelligence layer that (1) predicts when Basis Weight will go off-spec (>2.5%) *before* it happens, (2) recommends corrective setpoints, (3) reduces stabilization time, and (4) explains its reasoning.

---

## Our Solution

| Module | What it does |
|--------|--------------|
| **Forecast** | Predicts Basis Weight 5 min ahead (XGBoost) |
| **Off-spec risk** | Live 0–100% risk score; warns **~26 min early** |
| **Future-state projection** | Projects the trajectory: if the current trend continues → off-spec; with recommended setpoints → in-spec |
| **Recommendation** | Suggests safe setpoints from recipe + similar successful past runs; **every suggestion is source-tagged** (recipe / historical / model) |
| **Hidden-correlation discovery** | Surfaces **retention** — a driver of stabilization the standard control loops don't monitor |
| **Explainability** | SHAP explains *why* each alert fired |
| **Dashboard + feedback** | One-screen operator view; accept/reject suggestions are recorded for continuous learning |

---

## Key Results (on held-out test week)

- **Off-spec detection:** F1 = **0.94**, recall = **0.90**, ROC-AUC = **0.99**
- **Early warning:** median **~26 min** before a quality breach
- **Recommendation impact (example):** overshoot **16% → 2%**, stabilization **60 → 23 min** (~37 min saved)
- **Model selection:** XGBoost chosen via a **4-model comparison** (Linear, Random Forest, XGBoost, LightGBM), all on a **time-based split** (no data leakage)

---

## Repository Structure

```
Grade_Change_Intelligence/
├── README.md
├── notebooks/     # Jupyter notebooks (end-to-end pipeline)
├── src/           # Python scripts (data gen, models, dashboard, demo)
├── data/          # Generated datasets (CSV)
├── images/        # Graphs, dashboard & demo screenshots
└── deck/          # Idea presentation (PPTX + PDF)
```

### Main scripts (`src/`)
| File | Purpose |
|------|---------|
| `generate_data.py` | Physics-informed synthetic paper-machine historian |
| `p2_features.py` | Feature engineering (lags, rolling wobble, context) |
| `p2_5_reduce.py` | Redundant-feature removal + correlation EDA |
| `p3_forecast.py` | Basis Weight forecast model |
| `p3b_model_compare.py` | 4-model comparison |
| `p4_offspec.py` | Off-spec risk classifier |
| `p5_discovery.py` | Hidden-correlation (retention) discovery |
| `p6_similar.py` | Similar historical-case retrieval |
| `p7_recommend.py` | Recommendation engine (source-tagged) |
| `p8_explain.py` | SHAP explainability |
| `p9_feedback.py` | Operator feedback loop |
| `dashboard.py` | Streamlit operator dashboard |
| `realtime_demo.py` | Live what-if demo (interactive) |
| `demo_end_to_end.py` | Scripted end-to-end demo |

---

## How to Run

```bash
# 1. Install dependencies
pip install numpy pandas scikit-learn xgboost lightgbm shap matplotlib streamlit

# 2. Generate the dataset (creates paper_machine_data.csv)
python src/generate_data.py

# 3. Build features -> reduce -> train models (run in order)
python src/p2_features.py
python src/p2_5_reduce.py
python src/p3_forecast.py
python src/p4_offspec.py

# 4. Run the interactive live demo
streamlit run src/realtime_demo.py
```

> **Note:** the pipeline scripts read/write CSVs in the working directory. Keep the generated CSVs alongside the scripts (or adjust paths) when running the later stages.

---

## Approach Highlights (design decisions)

- **No production data was provided**, so we built a **physics-informed synthetic historian** with documented assumptions (recipe-based setpoints, actuator lag, transition overshoot, noise). The full pipeline runs unchanged on real plant data.
- **Time-based train/test split** (train on earlier weeks, test on the last) — never random — to avoid look-ahead leakage.
- **Interpretability first:** correlation-based feature pruning instead of PCA, so SHAP and the recommender keep real, human-readable variable names.
- **Hybrid recommendations:** historical good runs + recipe limits, verified by the model, with a **source tag on every suggestion**.

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · **XGBoost** · **LightGBM** · **SHAP** · **Streamlit** · Matplotlib

---

## References

- Honeywell Quality Control System (QCS) — MD Multivariable MPC grade change *(problem background)*
- T. Chen & C. Guestrin, *"XGBoost: A Scalable Tree Boosting System"*, KDD 2016
- S. Lundberg & S.-I. Lee, *"A Unified Approach to Interpreting Model Predictions"* (SHAP), NeurIPS 2017
- First-pass retention in papermaking — TAPPI wet-end chemistry concepts

---

## Note on Data

No production data was shared for this challenge. The dataset here is **synthetic but physics-informed**, created to demonstrate the methodology. Metrics reflect this controlled setting; the same models and pipeline apply directly to real QCS/DCS historian data.
