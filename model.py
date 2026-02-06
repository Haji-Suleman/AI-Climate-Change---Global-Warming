import torch
from cleaning import clean_Climate_Change_Dataset
from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = clean_Climate_Change_Dataset()
RANDOMSEED = 42
X = df.drop(columns=["climate_risk_index"]).values
y = df["climate_risk_index"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert into tensor
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


class Climate_Change(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=27, out_features=50),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=25),
            nn.ReLU(),
            nn.Linear(in_features=25, out_features=1),
        )

    def forward(self, X):
        return self.model(X)


torch.manual_seed(RANDOMSEED)
model_9 = Climate_Change()

print(model_9.state_dict())
