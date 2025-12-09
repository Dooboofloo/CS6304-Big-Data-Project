import os
import copy

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error

### ===== Data loading and preprocessing =====

# Read data into pandas dataframe
df = pd.read_csv("./uk-electricity-consumption-weather-combined.csv")

# Determine features and target to predict
FEATURES = ["year","month","day","hour","tmax_c","tmin_c","tavg_c","airfrost_days","rain_mm","sun_hrs","is_holiday"]

TARGET = ["nat_usage_mw"]

# Sort chronologically (should be so already, but just to make sure)
df = df.sort_values(["year","month","day","hour"]).reset_index(drop=True)

# Normalize all features except target
scaler_x = StandardScaler()
scaler_y = StandardScaler()

df[FEATURES] = scaler_x.fit_transform(df[FEATURES])
df[TARGET] = scaler_y.fit_transform(df[TARGET])


# ===== Set up important classes ======
class SlidingWindowDataset(Dataset):
    def __init__(self, df, window=720): # 720 hours = 30 days
        self.X = df[FEATURES].values.astype(np.float32)
        self.y = df[TARGET[0]].values.astype(np.float32)
        self.window = window
    
    def __len__(self):
        return len(self.y) - self.window
    
    def __getitem__(self, index):
        return (
            self.X[index:index + self.window],
            self.y[index + self.window]
        )

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d, dropout):
        super().__init__()
        
        padding = (k - 1) * d
        
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=padding, dilation=d)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=padding, dilation=d)
        self.norm = nn.LayerNorm(out_ch)
        self.dropout = nn.Dropout(dropout)

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        out = self.conv1(x)[:, :, :-self.conv1.padding[0]]
        out = self.dropout(F.relu(out))

        out = self.conv2(out)[:, :, :-self.conv2.padding[0]]
        out = self.dropout(F.relu(out))

        res = self.residual(x)
        return self.norm((out + res).transpose(1,2)).transpose(1,2)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()

        layers = []

        for i in range(len(num_channels)):
            dilation = 2 ** i

            in_ch = num_inputs if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]

            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        self.network = nn.ModuleList(layers)
        self.fc = nn.Linear(num_channels[-1], 1)
    
    def forward(self, x):
        x = x.transpose(1,2)
        
        for layer in self.network:
            x = layer(x)
        
        return self.fc(x[:, :, -1]).squeeze()


# ===== Training / Validation Split =====
dataset = SlidingWindowDataset(df, window=720) # 720 hours = 30 days

n = len(dataset)
train_end = int(0.8 * n) # 80% Training data
val_end = int(0.9 * n) # 10% Validation data (leaves 10% test data)

train_set = torch.utils.data.Subset(dataset, range(0, train_end))
val_set = torch.utils.data.Subset(dataset, range(train_end, val_end))
test_set = torch.utils.data.Subset(dataset, range(val_end, n))

# Set up loaders with settings for distributed computing
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=64, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=64, num_workers=4, pin_memory=True)


### ===== TRAINING LOOP =====
# This is the important stuff!

# Determine what kind of device this model will be training on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("USING DEVICE:", device)

# Set up the model, optimizer, and loss functions
model = TCN(
    num_inputs=len(FEATURES),
    num_channels=[64, 64, 64, 64],
    kernel_size=3,
    dropout=0.2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# Train the model with early stopping
epochs = 50
patience = 6
best_val_loss = float("inf")
best_model_state = None
epochs_no_improve = 0
best_epoch = 0

loss_history = []
val_loss_history = []

for epoch in range(epochs):
    # ===== Training =====
    model.train()
    total_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    loss_history.append(avg_train_loss)

    # ===== Validation =====
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            
            pred = model(X)
            loss = loss_fn(pred, y)
            val_loss += loss.item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_loss_history.append(avg_val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

    # ===== Early Stopping Check =====
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        epochs_no_improve = 0
        print("!!! Validation improved. Saving Best Model")
    else:
        epochs_no_improve += 1

        print(f"!!! No improvement for {epochs_no_improve} epoch(s)")

        if epochs_no_improve >= patience:
            print("!!! Early stopping triggered.")
            break

# Load back the best model state
model.load_state_dict(best_model_state)

# Save everything!
SAVE_DIR = "tcn_outputs"

# Save the model to disk
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "tcn_model.pt")
SCALER_X_PATH = os.path.join(SAVE_DIR, "scaler_x.npy")
SCALER_Y_PATH = os.path.join(SAVE_DIR, "scaler_y.npy")

torch.save({
    "best_epoch": best_epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict()
}, MODEL_PATH)

np.save(SCALER_X_PATH, np.vstack([scaler_x.mean_, scaler_x.scale_]))
np.save(SCALER_Y_PATH, np.vstack([scaler_y.mean_, scaler_y.scale_]))

print("!!! Model and scalers saved.")

# Save the training loss curve to disk
pd.DataFrame({"train_loss": loss_history, "val_loss": val_loss_history}).to_csv(
    "tcn_outputs/training_loss.csv", index=False
)
print("!!! Training loss history saved.")

### ===== Model Evaluation =====
# This is where we see if the model did well!

def evaluate(model, loader, output_csv="tcn_outputs/predictions.csv"):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()

            preds.extend(pred)
            trues.extend(y.numpy())
    
    preds = scaler_y.inverse_transform(np.array(preds).reshape(-1,1)).flatten()
    trues = scaler_y.inverse_transform(np.array(trues).reshape(-1,1)).flatten()

    # Metrics
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    wape = np.sum(np.abs(trues-preds)) / np.sum(np.abs(trues))

    # Save evaluation results to disk
    results_df = pd.DataFrame({
        "true_usage_mw": trues,
        "predicted_usage_mw": preds,
        "absolute_error": np.abs(preds - trues)
    })

    results_df.to_csv(output_csv, index=False)

    print(f"!!! Predictions saved to {output_csv}.")

    return rmse, mae, wape

rmse, mae, wape = evaluate(model, test_loader)

# Print results
print("RMSE:", rmse)
print("MAE:", mae)
print("WAPE:", wape)
print("Best Epoch:", best_epoch)

metrics_df = pd.DataFrame([{
    "RMSE": rmse,
    "MAE": mae,
    "WAPE": wape
}])

metrics_output = "tcn_outputs/final_metrics.csv"

metrics_df.to_csv(metrics_output, index=False)

print(f"!!! Final metrics saved to {metrics_output}")
