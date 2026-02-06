import pandas as pd
from sklearn.preprocessing import StandardScaler


def getting_Model_Ready(df: pd.DataFrame):
    X = df.drop(columns=["id", "climate_risk_index"]).values
    y = df["climate_risk_index"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert into tensor
    
