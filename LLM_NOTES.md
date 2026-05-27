# LLM / AI Tool Usage Notes

## Tools Used

- ChatGPT

---

# Where AI Tools Were Used

- pipeline improvement
- preprocessing suggestions
- auditability improvements
- documentation drafting

---

# What I Accepted

1. Using sklearn Pipeline and ColumnTransformer for consistent preprocessing.

2. Replacing LabelEncoder with OneHotEncoder for categorical features.

3. Removing leakage features and separating policy logic from the model.

4. Adding artifact saving, schema validation, and reason-code generation.

---

# What I Corrected

1. Removed future leakage information from training.

2. Removed direct score manipulation based on PM-Kisan status.

3. Improved validation approach for safer scoring simulation.

---

# What I Personally Verified

- pipeline execution
- validation output
- artifact generation
- fairness monitoring
- reason-code generation
- schema validation

---

# Personal Responsibility Statement

All code, outputs, and documentation were reviewed and verified before final submission.