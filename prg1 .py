
# ============================================================
# MINI PROJECT: HEART DISEASE PREDICTION
# Logistic Regression + Evaluation + Visualization
# ============================================================

# ------------------------------------------------------------
# 1. Import required libraries
# ------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
)


# ------------------------------------------------------------
# 2. Load the Dataset
# ------------------------------------------------------------

# Dataset Name: Heart Disease Dataset
# File name: heart.csv

df = pd.read_csv("heart.csv")

print("Dataset Name: Heart Disease Dataset")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)


# ------------------------------------------------------------
# 3. Data Cleaning and Target Preparation
# ------------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# The dataset does not contain a 'target' column.
# The 'num' column indicates presence/absence of heart disease.
# Convert to binary classification: 0 = healthy, 1 = disease
# (values > 0 represent different stages of heart disease)
df["target"] = (df["num"] > 0).astype(int)

print("\nShape after removing duplicates:")
print(df.shape)
print("\nTarget distribution:")
print(df["target"].value_counts())


# ------------------------------------------------------------
# 4. Separate Features and Target
# ------------------------------------------------------------

X = df.drop(columns=["target", "num", "id"], errors="ignore")
y = df["target"]

# Fill missing numeric values with median and encode categorical columns
X = X.copy()

for col in X.select_dtypes(include="number"):
    X[col] = X[col].fillna(X[col].median())

for col in X.select_dtypes(include="object"):
    X[col] = X[col].fillna("Unknown")

X = pd.get_dummies(X, drop_first=True)


# ------------------------------------------------------------
# 5. Split Dataset into Training and Testing Data
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------------------
# 6. Feature Scaling
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 7. Train Logistic Regression Model
# ------------------------------------------------------------

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)


# ------------------------------------------------------------
# 8. Make Predictions
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)
y_probability = model.predict_proba(X_test_scaled)[:, 1]


# ------------------------------------------------------------
# 9. Calculate Evaluation Metrics
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)

print("\n======================================")
print("       MODEL EVALUATION")
print("======================================")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))


# ------------------------------------------------------------
# 10. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
confusion_display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Disease", "Heart Disease"],
)

confusion_display.plot()
plt.title("Confusion Matrix")
plt.show()


# ------------------------------------------------------------
# 11. ROC Curve
# ------------------------------------------------------------

fpr, tpr, _ = roc_curve(y_test, y_probability)
auc_score = roc_auc_score(y_test, y_probability)

print("\nROC-AUC Score:", round(auc_score, 4))

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="Logistic Regression")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ------------------------------------------------------------
# 12. Compare Actual vs Predicted Values
# ------------------------------------------------------------

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
})

print("\nActual vs Predicted:")
print(comparison.head(20))


# ------------------------------------------------------------
# 13. Final Result
# ------------------------------------------------------------

print("\n======================================")
print("        PROJECT COMPLETED")
print("======================================")

print("Dataset       : Heart Disease Dataset")
print("Model         : Logistic Regression")
print("Features      :", X.shape[1])
print("Accuracy      :", round(accuracy, 4))
print("Precision     :", round(precision, 4))
print("Recall        :", round(recall, 4))
print("ROC-AUC       :", round(auc_score, 4))

