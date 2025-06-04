import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("education_binary_classification.csv")
X = df.drop(columns=['target'])
y = df['target']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression with class balancing
model = LogisticRegression(C=1.0, solver='liblinear', class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Predict probabilities
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# Set static threshold
threshold = 0.5  # <-- Промени тук, ако искаш друг праг
print(f"Using static threshold: {threshold}")

# Apply threshold
y_pred = (y_proba >= threshold).astype(int)

# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = conf_matrix.ravel()

# Manual accuracy
manual_accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"Manual Accuracy: {manual_accuracy:.4f}")

# Metrics
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

print(f"Precision: {precision:.4f}")
print(f"Recall (TPR): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"False Positive Rate (FPR): {FP / (FP + TN):.4f}")
print(f"False Negative Rate (FNR): {FN / (FN + TP):.4f}")

# Confusion matrix heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# ROC Curve
fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
