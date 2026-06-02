import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import os


def preprocess_data(input_path, output_path):

    print("Current working directory:")
    print(os.getcwd())

    print("\nFiles in current directory:")
    print(os.listdir())

    df = pd.read_csv(input_path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"].fillna(
        df["TotalCharges"].median(),
        inplace=True
    )

    df.drop_duplicates(inplace=True)

    df.drop(columns=["customerID"], inplace=True)

    le = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()

    numerical_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    df[numerical_cols] = scaler.fit_transform(
        df[numerical_cols]
    )

    df.to_csv(output_path, index=False)

    print("Preprocessing selesai")


if __name__ == "__main__":

    preprocess_data(
    "Eksperimen_SML_Telco_Churn/Telco-Customer-Churn.csv",
    "Eksperimen_SML_Telco_Churn/preprocessing/telco_preprocessing.csv"
    )