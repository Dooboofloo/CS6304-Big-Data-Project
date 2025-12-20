# 📈 Long-Term Electricity Consumption Forecasting with Temporal Convolutional Networks

**Developer:** Jerrett Martin

This project explores **long-term electricity demand forecasting** using **Temporal Convolutional Networks (TCNs)** and large-scale time-series data.

The model integrates **weather, temporal, and calendar features** and demonstrates strong performance on real-world UK electricity consumption data.

---

## 🔎 Project Overview

Accurate electricity demand forecasting is essential as energy systems face increasing strain from electrification, climate variability, and emerging technologies (e.g., data centers, AI workloads).

This project investigates whether **TCNs — using causal, dilated convolutions — can outperform traditional ML and recurrent models** when forecasting **long-range consumption trends**.

### Key Contributions
- Designed and implemented a **custom TCN architecture** in PyTorch
- Integrated **exogenous variables** (weather + holidays) into the time-series pipeline
- Applied **sliding-window segmentation** to reduce overfitting on long horizons
- Used **chronological data splits** to prevent temporal leakage
- Trained and evaluated the model on **~279,000 semi-hourly observations**
- Achieved **sub-10% error** on long-term forecasts

---

## 🧠 Model Architecture

The model is a **Temporal Convolutional Network (TCN)** featuring:

- **Causal 1D convolutions** (strictly past-dependent predictions)
- **Dilated convolutions** with exponentially increasing receptive fields
- **Residual connections** to stabilize deep temporal learning
- **Layer normalization & spatial dropout** to improve generalization
- **Sliding window inputs** (15-day / 720-half-hour windows)

### Architecture Summary
- **Input:** 11 features × 720 timesteps
- **Channels:** `[64, 64, 64, 64]`
- **Kernel size:** `3`
- **Dilation schedule:** `[1, 2, 4, 8]`
- **Dropout:** `0.3`
- **Optimizer:** Adam (`1e-3`)
- **Loss:** Mean Squared Error (MSE)
- **Early stopping:** patience = 6 epochs

---

## 🗂️ Dataset

The model is trained on a merged dataset consisting of:

- **UK National Electricity Consumption (2009–2024)**  
  Hourly electricity usage in megawatts (MW)

- **UK Met Office Historic Weather Data**
  - Max / Min / Avg temperature
  - Rainfall
  - Sunlight hours
  - Air frost days

- **Calendar features**
  - Year, month, day, hour
  - Binary holiday indicator

All features are standardized using **training-set-only statistics** to prevent leakage.

---

## 📊 Results

The final model was evaluated on a held-out test set of unseen future data.

| Metric | Value |
|------|------|
| **RMSE** | **2,912 MW** |
| **MAE** | **2,252 MW** |
| **WAPE** | **8.83%** |

These results indicate:
- Predictions deviate **< 9% on average** from true consumption
- Errors remain well below thresholds associated with catastrophic over/under-provisioning
- The model generalizes effectively despite strong seasonal and economic variability

Early stopping successfully prevented overfitting, with the best model selected at epoch 2.

---

## 🚀 How to Run

### Requirements
- Python 3.9+
- PyTorch
- NumPy, Pandas, scikit-learn

### Training & Evaluation
```bash
python main.py