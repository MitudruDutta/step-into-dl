# 📊 Evaluation Metrics

Choosing the right metrics helps you understand how well your model performs and where it needs improvement.

---

## Classification Metrics

### Core Concepts

Before diving into metrics, understand these terms:

| Term | Definition |
|------|------------|
| **True Positive (TP)** | Correctly predicted positive |
| **True Negative (TN)** | Correctly predicted negative |
| **False Positive (FP)** | Incorrectly predicted positive (Type I error) |
| **False Negative (FN)** | Incorrectly predicted negative (Type II error) |

### Confusion Matrix

```
                    Predicted
                 Positive  Negative
Actual Positive    TP        FN
Actual Negative    FP        TN
```

---

### Key Metrics

| Metric | Formula | When to Use |
|--------|---------|-------------|
| **Accuracy** | (TP + TN) / Total | Balanced datasets |
| **Precision** | TP / (TP + FP) | When false positives are costly |
| **Recall** | TP / (TP + FN) | When false negatives are costly |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Imbalanced datasets |


### Choosing the Right Metric

**Use Accuracy when**:
- Classes are balanced
- All errors are equally costly

**Use Precision when**:
- False positives are expensive
- Example: Spam detection (don't want to miss important emails)

**Use Recall when**:
- False negatives are expensive
- Example: Cancer detection (don't want to miss actual cases)

**Use F1 Score when**:
- Classes are imbalanced
- You need balance between precision and recall

---

## Regression Metrics

| Metric | Description | Characteristics |
|--------|-------------|-----------------|
| **MAE** | Mean Absolute Error | Robust to outliers; same units as target |
| **MSE** | Mean Squared Error | Penalizes large errors more heavily |
| **RMSE** | Root Mean Squared Error | Same units as target; penalizes large errors |
| **R² Score** | Coefficient of determination | 1.0 = perfect; proportion of variance explained |

### When to Use Each

- **MAE**: When outliers should have less influence
- **MSE/RMSE**: When large errors are particularly bad
- **R²**: When you want to understand how much variance is explained

---

## Multi-Class Metrics

For problems with more than two classes:

- **Macro Average**: Calculate metric for each class, then average (treats all classes equally)
- **Micro Average**: Aggregate TP, FP, FN across all classes, then calculate (weighted by class frequency)
- **Weighted Average**: Like macro, but weighted by class support

---

*Choose metrics that align with your business objectives. Accuracy alone is often misleading for imbalanced datasets.*
