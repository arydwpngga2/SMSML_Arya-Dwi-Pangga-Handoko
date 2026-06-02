import pandas as pd
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Telco_Churn")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# aktifkan autolog
mlflow.sklearn.autolog()

# load data
df = pd.read_csv("telco_preprocessing.csv")

# fitur dan target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# model
model = RandomForestClassifier(
    random_state=42
)

# training
model.fit(X_train, y_train)

# prediksi
y_pred = model.predict(X_test)

# evaluasi
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")