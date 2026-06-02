import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn

dagshub.init(
    repo_owner="arydwpngga2",
    repo_name="telco-churn-mlops",
    mlflow=True
)

# Tracking MLflow
mlflow.set_experiment("Telco_Churn_Tuning")

print("Tracking URI:", mlflow.get_tracking_uri())

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("telco_preprocessing.csv")

X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Prediksi
y_pred = best_model.predict(X_test)

# Evaluasi
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Best Parameters:")
print(grid_search.best_params_)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

# Classification Report
with open("classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred))

# Manual Logging
with mlflow.start_run():

    mlflow.log_param(
        "n_estimators",
        best_model.n_estimators
    )

    mlflow.log_param(
        "max_depth",
        best_model.max_depth
    )

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    mlflow.log_artifact(
        "confusion_matrix.png"
    )

    mlflow.log_artifact(
        "classification_report.txt"
    )

    mlflow.sklearn.log_model(
        best_model,
        "model"
    )