# Audit Memo - Yogyank Entitlement Score

## Overview

The original baseline training script was unsafe for real-world deployment because it contained multiple critical ML engineering and governance issues. Although the model reported a very high validation R² score, the result was not trustworthy due to severe data leakage and policy contamination.

---

# Key Problems Identified

## 1. Data Leakage

The original script used the feature:

- defaulted_in_next_12_months

This feature contains future information that would not be available during real scoring. Using future outcome information caused unrealistically high validation performance and made the model unreliable for production use.

---

## 2. Policy Logic Mixed Into Model Training

The original script directly modified the target entitlement score for farmers without PM-Kisan status.

This is dangerous because business policy decisions should remain outside the ML model. The model should predict entitlement independently, while downstream policy decisions should be handled separately in a versioned policy layer.

---

## 3. Unsafe Encoding Strategy

The original implementation used LabelEncoder on nominal categorical variables. This imposed artificial numeric ordering on categories and could distort model behavior.

---

## 4. Weak Validation Design

The dataset was randomly shuffled before splitting into train and validation sets. This does not realistically simulate future scoring conditions and may hide leakage risks.

---

## 5. Missing Auditability

The original script only saved the trained model and did not save:
- preprocessing pipeline
- feature schema
- metadata
- version information
- explanation artifacts

This reduced reproducibility and governance quality.

---

# Improvements Implemented

## Leakage Prevention

Removed the leakage feature:
- defaulted_in_next_12_months

---

## Separation of Policy and Model Logic

Removed direct manipulation of target scores based on PM-Kisan status.

---

## Safer Preprocessing

Implemented:
- sklearn Pipeline
- ColumnTransformer
- OneHotEncoder
- missing value handling

This ensures consistent preprocessing during inference.

---

## Improved Validation

Used deterministic validation split with:
- shuffle=False

to better simulate future scoring behavior.

---

## Auditability Improvements

Added:
- saved model pipeline
- feature schema
- metadata JSON
- artifact manifest
- version information

---

## Explainability

Implemented top reason-code generation using feature importance rankings.

---

# Remaining Limitations

## Limitation 1

The dataset is synthetic and relatively small, so validation metrics may not fully represent real-world performance.

## Limitation 2

The current implementation uses global feature importance instead of advanced local explanation methods such as SHAP.

---

# Future Improvements

With additional time, I would:
- implement out-of-time validation
- add drift monitoring
- add fairness metrics
- implement SHAP explanations
- add automated testing

---

# Monitoring Recommendations

After deployment, I would monitor:
- crop type performance
- PM-Kisan status fairness
- district-wise score stability
- landholding bands
- irrigation-type stability