import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from typing import Tuple


def getting_Model_Ready(df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    RANDOMSEED = 42
    X = df.drop(columns=["id", "climate_risk_index"]).values
    y = df["climate_risk_index"].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert into tensor
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    return X, y
