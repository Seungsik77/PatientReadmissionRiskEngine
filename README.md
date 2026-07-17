# Patient Readmission Risk Engine

An end-to-end clinical ML pipeline predicting 30-day hospital readmission using structured EHR data. Built with pandas, scikit-learn, and PyTorch — including probability calibration and Monte Carlo Dropout uncertainty quantification.

---

## Overview

Hospital readmissions within 30 days cost the US healthcare system billions annually. CMS penalizes institutions up to 3% of Medicare payments for excess readmission rates under the Hospital Readmissions Reduction Program (HRRP). This project builds a risk stratification engine that flags high-risk patients at discharge — and quantifies how confident the model is in each prediction.

**Key results on MIMIC-IV Demo:**
- Calibrated Gradient Boosting AUC: **0.764**
- PyTorch deep model AUC: **0.528** (demo dataset too small for deep model advantage — expected to close on full MIMIC-IV)
- MC Dropout uncertainty flags ambiguous patients for clinician review

---

## Pipeline

```
Raw EHR tables (MIMIC-IV)
        │
        ▼
[ pandas ] ── cohort construction, feature engineering, Charlson index
        │
        ▼
[ scikit-learn ] ── preprocessing pipeline, calibrated GBM baseline, SHAP
        │
        ▼
[ PyTorch ] ── feedforward net + MC Dropout uncertainty quantification
        │
        ▼
risk score + uncertainty flag per patient
```

---

## Dataset

**MIMIC-IV Clinical Database Demo** (PhysioNet)

Place the extracted files at:
```
PatientReadmissionRiskEngine/
└── hosp/
    ├── admissions.csv
    ├── patients.csv
    └── diagnoses_icd.csv
```

---

## Features Engineered

| Feature | Source | Description |
|---|---|---|
| `los_days` | admissions | Length of current stay in days |
| `anchor_age` | patients | Patient age at admission |
| `prior_visit_count` | admissions | Number of prior admissions (cumulative) |
| `charlson_index` | diagnoses_icd | Simplified Charlson Comorbidity Index from ICD codes |
| `admission_type` | admissions | Categorical admission type |
| `insurance` | admissions | Payer type |
| `gender` | patients | Patient sex |

**Label:** `readmitted_30d` — binary flag, 1 if next admission occurs within 30 days of discharge. Measured from `dischtime` to next `admittime` for the same patient. Final admission per patient is excluded (no follow-up window).

---

## Models

### Baseline — Calibrated Gradient Boosting (scikit-learn)

- `ColumnTransformer`: `StandardScaler` on numerics, `OneHotEncoder` on categoricals
- `GradientBoostingClassifier` (n_estimators=100)
- `CalibratedClassifierCV` with isotonic regression reduces ECE
- SHAP `TreeExplainer` for feature importance
- Top features by SHAP: `los_days`, `anchor_age`, `prior_visit_count`, `charlson_index`

### Deep Model — ReadmissionNet (PyTorch)

```
Input (4) → Linear(4→32) → ReLU → Dropout(0.3)
          → Linear(32→16) → ReLU → Dropout(0.3)
          → Linear(16→1) → Sigmoid
```

- `WeightedRandomSampler` handles class imbalance (~70/30 split)
- Adam optimizer, lr=1e-3, 100 epochs
- **MC Dropout:** 30 stochastic forward passes at inference with `model.train()` kept active
  - `mean(30 passes)` → readmission probability
  - `std(30 passes)` → epistemic uncertainty
  - Patients with `std > 0.12` flagged for clinician review

---


## Results

| Model | AUC | Results |
|---|---|---|
| Logistic Regression | 0.525 | Simple linear baseline |
| Gradient Boosting (calibrated) | 0.764 | Primary baseline |
| ReadmissionNet + MC Dropout | 0.528 | Limited by demo dataset size |

> On the 100-patient demo, calibrated Gradient Boosting outperforms the PyTorch model — consistent with literature showing tree-based methods dominate on small tabular datasets. The neural network architecture is designed to scale to full MIMIC-IV (500K admissions) where deep models recover their advantage.

---

## Key Design Decisions

**Patient-level train/test split** — rows are split by `subject_id`, not randomly, to prevent data leakage from the same patient appearing in both train and test sets.

**Probability calibration** — raw GBM probabilities are poorly calibrated. Isotonic regression via `CalibratedClassifierCV` produces reliable probability outputs suitable for clinical thresholds.

**MC Dropout for uncertainty** — keeping `model.train()` active at inference time means dropout fires on every forward pass. Running 30 passes and taking the standard deviation gives an epistemic uncertainty estimate. High-variance predictions are flagged rather than acted on automatically.

**Charlson Comorbidity Index** — derived from ICD-10 diagnosis codes using simplified prefix matching. Encodes disease burden in a single interpretable feature that clinicians recognize.

---

## Acknowledgements

Data from the MIMIC-IV Clinical Database Demo, PhysioNet. Johnson, A., Bulgarelli, L., Pollard, T., Horng, S., Celi, L. A., & Mark, R. (2023). MIMIC-IV Clinical Database Demo (version 2.2). PhysioNet.
