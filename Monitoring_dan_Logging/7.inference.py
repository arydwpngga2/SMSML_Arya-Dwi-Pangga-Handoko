import joblib
import pandas as pd

model = joblib.load("../Membangun_model/mlruns/2/models/m-1eb193fab24549de8a3b64cc06d25eb2/artifacts/model.pkl")

df = pd.read_csv("../Membangun_model/telco_preprocessing.csv")

X = df.drop("Churn", axis=1)

sample = X.iloc[[0]]

pred = model.predict(sample)

print("Prediction:", pred)
print("Jumlah fitur:", len(sample.columns))