import torch
from cleaning import clean_Climate_Change_Dataset
from sklearn.model_selection import train_test_split
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load and prepare data
df = clean_Climate_Change_Dataset()
RANDOMSEED = 42
X = df.drop(columns=["climate_risk_index"]).values
y = df["climate_risk_index"].values.reshape(-1, 1)

# Scale features and target
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOMSEED
)


# Model
class Climate_Change(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(27, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, X):
        return self.model(X)


torch.manual_seed(RANDOMSEED)
model = Climate_Change()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training
epochs = 2000
train_losses = []
test_losses = []

for epoch in range(epochs):
    model.train()
    y_pred_train = model(X_train)
    loss = loss_fn(y_pred_train, y_train)
    optimizer.zero_grad()   
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_pred_test = model(X_test)
        test_loss = loss_fn(y_pred_test, y_test)

    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    if epoch % 200 == 0:
        print(
            f"Epoch {epoch}: Train Loss={loss.item():.4f}, Test Loss={test_loss.item():.4f}"
        )

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Train/Test Loss")
plt.legend()
plt.show()

# Predictions vs Actual (inverse scaled)
y_pred_all = scaler_y.inverse_transform(model(X).detach().numpy())
y_actual = scaler_y.inverse_transform(y.numpy())

plt.figure(figsize=(10, 5))
plt.plot(y_actual, label="Actual", marker="o", linestyle="")
plt.plot(y_pred_all, label="Predicted", marker="x", linestyle="")
plt.xlabel("Samples")
plt.ylabel("Climate Risk Index")
plt.title("Actual vs Predicted Climate Risk Index")
plt.legend()
plt.show()

# R² score (good metric for regression)
r2 = r2_score(y_actual, y_pred_all)
print(f"R² Score: {r2:.4f}")
