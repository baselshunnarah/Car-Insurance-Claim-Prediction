[README (1).md](https://github.com/user-attachments/files/26031697/README.1.md)
# 🚗 Car Insurance Claim Prediction

A machine learning project that predicts whether a customer will file a car insurance claim based on driver profile, vehicle information, and behavioral history.

---

## 📌 Project Overview

**Goal:** Binary classification — predict `outcome` (1 = claim filed, 0 = no claim).

**Business Priority:** Missing a high-risk driver (**False Negative**) is more costly than misidentifying a low-risk one (**False Positive**). For this reason, we prioritize **Recall, Precision, and F1-Score** over raw Accuracy when selecting and evaluating models.

---

## 📂 Dataset

| Feature | Type | Description |
|---|---|---|
| `AGE` | Categorical | Age group of the driver (16-25, 26-39, 40-64, 65+) |
| `GENDER` | Binary | Driver's gender |
| `DRIVING_EXPERIENCE` | Ordinal | Years of experience (0-9y, 10-19y, 20-29y, 30y+) |
| `EDUCATION` | Ordinal | Highest education level |
| `INCOME` | Ordinal | Income bracket (poverty → upper class) |
| `CREDIT_SCORE` | Float | Normalized credit score (0–1) |
| `VEHICLE_OWNERSHIP` | Binary | Owns vehicle (1) or not (0) |
| `VEHICLE_YEAR` | Categorical | Before/After 2015 |
| `MARRIED` | Binary | Marital status |
| `CHILDREN` | Binary | Has children |
| `POSTAL_CODE` | Integer | Driver's location |
| `ANNUAL_MILEAGE` | Float | Estimated annual miles driven |
| `VEHICLE_TYPE` | Categorical | Sedan, SUV, Sports Car, etc. |
| `SPEEDING_VIOLATIONS` | Integer | Number of speeding violations |
| `DUIS` | Integer | Number of DUI offenses |
| `PAST_ACCIDENTS` | Integer | Number of past accidents |
| `OUTCOME` | Binary | **Target** — 1 = claim filed, 0 = no claim |

**Dataset size:** 10,000 records (train/test split applied)

---

## ⚙️ Methodology (CRISP-DM)

1. **Business Understanding** — Frame as a risk classification problem with asymmetric costs (FN > FP).
2. **Data Understanding** — EDA revealed that age group `16-25` and low driving experience are the strongest predictors of claims.
3. **Data Preparation**
   - Ordinal encoding for `DRIVING_EXPERIENCE`, `EDUCATION`, `INCOME`
   - One-hot encoding for nominal categoricals (`AGE`, `GENDER`, `VEHICLE_TYPE`, etc.)
   - StandardScaler for numeric features
   - Median imputation for missing values
   - SMOTE oversampling to handle class imbalance on training data
4. **Modeling** — Multiple models trained and compared (see below).
5. **Evaluation** — Models evaluated on a held-out test set (2,500 samples) using Precision, Recall, and F1-Score for the positive class (high-risk drivers).
6. **Feature Engineering** — PCA (3 components) combined with original features; collinearity filtering applied.

---

## 🤖 Models Trained

| Model | Test Accuracy | Precision (Class 1) | Recall (Class 1) | F1 (Class 1) |
|---|---|---|---|---|
| Logistic Regression | 82% | 0.67 | 0.84 | 0.74 |
| Random Forest (baseline) | 84% | 0.72 | 0.77 | 0.75 |
| Random Forest + PCA | 84% | 0.75 | 0.73 | 0.74 |
| **Random Forest + Feature Selection** ✅ | **84%** | **0.72** | **0.79** | **0.75** |
| Neural Network (base) | 83% | 0.74 | 0.73 | 0.73 |
| Neural Network (Keras Tuner) | 83% | 0.75 | 0.68 | 0.72 |

---

## 🏆 Best Model

**Random Forest Classifier** with collinearity-filtered features (PCA components + original features, correlated features removed).

```python
RandomForestClassifier(max_depth=10, n_estimators=200)
```

### Test Set Performance (2,500 unseen samples)

| | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Class 0 (No Claim) | 0.90 | 0.86 | 0.88 | 1,723 |
| **Class 1 (Claim / High-Risk)** | **0.72** | **0.79** | **0.75** | 777 |
| **Overall Accuracy** | | | **84%** | 2,500 |

### Why This Model?

Given our business constraint — **False Negatives are more expensive** — Recall on the positive class is the deciding metric. The Random Forest with feature selection achieves:

- **Recall = 0.79**: Correctly flags 79% of actual high-risk drivers, the highest recall among all models.
- **F1-Score = 0.75**: Best balance between precision and recall on Class 1.
- **Precision = 0.72**: 72% of predicted high-risk drivers are genuinely high-risk — acceptable false alarm rate.

The tuned Neural Network achieved higher precision (0.75) but suffered on recall (0.68), meaning it missed more high-risk drivers — which is the more costly error in this domain.

---

## 🔑 Key Findings

- **Age** and **Driving Experience** are the top two predictors of claims, confirming EDA insights. Drivers aged 16–25 have the highest actual claim rate (~75%).
- **Past accidents**, **speeding violations**, and **DUIs** are strong behavioral risk indicators.
- All models plateaued at ~84% accuracy due to the inherent difficulty of the task, but differ meaningfully in how they handle the minority (high-risk) class.
- Mild overfitting was observed across tree-based models (train accuracy ~91–92% vs. test ~84%).



