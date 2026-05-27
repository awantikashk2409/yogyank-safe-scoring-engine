# Yogyank Safe Baseline Pipeline

## Author

Awantika Srivastava

---

# Objective

The objective of this assessment is to improve the unsafe baseline training script into a safer, leakage-aware, reproducible, and audit-friendly machine learning pipeline suitable for regulated scoring environments.

The focus of this solution is not maximizing model accuracy, but improving:
- leakage prevention
- validation reliability
- auditability
- explainability
- reproducibility
- fairness awareness
- separation of model logic from business policy

---

# Summary of Problems in Original Script

The original baseline implementation contained multiple critical issues:

## 1. Data Leakage

The feature:

- `defaulted_in_next_12_months`

was used during training. This feature contains future information that would not be available at scoring time, leading to unrealistically high validation performance.

---

## 2. Policy Contamination

The original script directly modified the target score based on PM-Kisan status before training.

This mixes business policy with ML model behavior and creates bias inside the model itself.

---

## 3. Unsafe Encoding

The original implementation used `LabelEncoder` on nominal categorical variables, which imposed artificial numeric ordering on categories.

---

## 4. Weak Validation Strategy

The dataset was randomly shuffled before splitting into train and validation sets, which does not realistically simulate future scoring scenarios.

---

## 5. Missing Auditability

The original script only saved the trained model and did not save:
- preprocessing artifacts
- feature schema
- metadata
- version information
- reason-code logic

This reduced reproducibility and governance quality.

---

# Improvements Implemented

## Leakage Prevention

Removed:
- `defaulted_in_next_12_months`

from the feature set.

---

## Separation of Policy and Model Logic

Removed all direct manipulation of the target score based on PM-Kisan status.

---

## Safer Preprocessing Pipeline

Implemented:
- `Pipeline`
- `ColumnTransformer`
- `OneHotEncoder`
- missing value handling

to ensure consistent preprocessing during training and inference.

---

## Improved Validation

Used deterministic validation split with:
- `shuffle=False`
- fixed random state

to better simulate future scoring behavior.

---

## Fairness Monitoring

Added monitoring slices for:
- PM-Kisan status
- crop type

to support fairness and stability analysis.

---

## Explainability

Implemented top reason-code generation using model feature importance rankings.

---

## Auditability Improvements

Saved:
- full training pipeline
- feature schema
- metadata
- artifact manifest
- version information

for reproducible scoring.

---

# Model Used

- RandomForestRegressor

Reason for selection:
- stable and reproducible
- easy to explain
- strong baseline performance
- simpler dependency management
- suitable for audit-focused environments

---


# Setup Instructions

Install required dependencies:

pip install pandas numpy scikit-learn joblib