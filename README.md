# 📈 Long-Term Electricity Consumption Forecasting with Temporal Convolutional Networks

**Developer:** Jerrett Martin

This project explores **long-term electricity demand forecasting** using **Temporal Convolutional Networks (TCNs)** and multi-year time-series data.

The model integrates **weather, temporal, and calendar features** and demonstrates strong performance on real-world UK electricity consumption data.

It was developed as a graduate project for Missouri S&T's CS 6304 — Cloud Computing and Big Data Management.

---

## 🔎 Project Overview

Accurate electricity demand forecasting is essential as energy systems face increasing strain from electrification, climate variability, and emerging technologies (e.g., data centers, AI workloads).

This project investigates the effectiveness of **TCNs — using causal, dilated convolutions — for forecasting electricity consumption over horizons spanning months to years.**

### Key Contributions
- Designed and implemented a **custom TCN architecture** in PyTorch
- Integrated **exogenous variables** (weather + holidays) into the time-series pipeline
- Applied **sliding-window segmentation** to reduce overfitting on long horizons
- Used **chronological data splits** to prevent temporal leakage
- Trained and evaluated the model on **~279,000 semi-hourly observations**
- Achieved **8.83% WAPE** on a chronologically held-out future test set

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
  Half-hourly electricity usage in megawatts (MW)

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
- The model maintained **sub-10% aggregate forecast error** on chronologically held-out future data
- The model generalized across a multi-year electricity consumption dataset with strong seasonal variation

Early stopping was used to select the best validation checkpoint and limit overfitting.

---

## 🚀 How to Run

### Requirements
- Python 3.9+
- PyTorch
- NumPy, Pandas, scikit-learn

### Training & Evaluation
```bash
python main.py
```

## 📌 Project Status

This project is **feature-complete and archived**.  
No further development is planned, and the repository is preserved in its final state for reference and evaluation.
