import torch
from cleaning import clean_Climate_Change_Dataset
from sklearn.model_selection import train_test_split
from torch import nn
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = clean_Climate_Change_Dataset()
RANDOMSEED = 42
X = df.drop(columns=["climate_risk_index"]).values
y = df["climate_risk_index"].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOMSEED
)
y_train = y_train.unsqueeze(1)
y_test = y_test.unsqueeze(1)


class Climate_Change(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=27, out_features=50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

    def forward(self, X):
        return self.model(X)


torch.manual_seed(RANDOMSEED)
model_9 = Climate_Change()

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model_9.parameters(), lr=0.001)

epochs = 1000
train_losses = []
test_losses = []

for epoch in range(epochs):
    model_9.train()
    y_pred_train = model_9(X_train)
    loss = loss_fn(y_pred_train, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_9.eval()
    with torch.inference_mode():
        y_pred_test = model_9(X_test)
        test_loss = loss_fn(y_pred_test, y_test)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss:.4f}, Test Loss = {test_loss:.4f}")

# Plot Train/Test Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Train and Test Loss over Epochs")
plt.legend()
plt.show()

# Plot Predictions vs Actual
y_pred_all = model_9(X).detach().numpy()
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y)), y.numpy(), label="Actual Climate Risk Index", alpha=0.7)
plt.scatter(
    range(len(y_pred_all)), y_pred_all, label="Predicted Climate Risk Index", alpha=0.7
)
plt.xlabel("Sample")
plt.ylabel("Climate Risk Index")
plt.title("Actual vs Predicted Climate Risk Index")
plt.legend()
plt.show()
