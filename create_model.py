"""
Run this script ONCE inside your venv311_correct to generate the model file.

    venv311_correct\Scripts\activate
    pip install tensorflow yfinance scikit-learn numpy pandas
    python create_model.py

It will save  stock_sentiment_model.pt.h5  in the same folder.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("📥 Downloading AAPL data (2011–2019)...")
df = yf.download("AAPL", start="2011-02-01", end="2019-12-31",
                 auto_adjust=True, progress=False)

# Flatten MultiIndex if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index().dropna()
close = df["Close"].values.reshape(-1, 1)

# ── Scale ─────────────────────────────────────────────────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(close)

# ── Build sequences (window = 100 days) ───────────────────────────────────────
WINDOW = 100
X, y = [], []
for i in range(WINDOW, len(scaled)):
    X.append(scaled[i - WINDOW:i, 0])
    y.append(scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape(X.shape[0], X.shape[1], 1)   # (samples, timesteps, features)

# ── Train / test split (70 / 30) ─────────────────────────────────────────────
split = int(len(X) * 0.70)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"✅ Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(WINDOW, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1),
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\n🚀 Training… (this takes ~2–5 minutes on CPU)")
model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1,
)

# ── Save ──────────────────────────────────────────────────────────────────────
MODEL_PATH = "stock_sentiment_model.pt.h5"
model.save(MODEL_PATH)
print(f"\n✅ Model saved as:  {MODEL_PATH}")
print("👉 Copy this file into your project folder and restart the Streamlit app.")
