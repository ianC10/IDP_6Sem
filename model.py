import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. LOAD DATA
df = pd.read_csv('FINAL_TRAINING_DATA.csv')

# Features: IMU (Z-accel, Y-gyro), Master Force, and Master Knee Angle
features = ['accel_z', 'gyro_y', 'force_z', 'master_knee']
target = 'target_knee'

X_raw = df[features].values
y_raw = df[target].values

# 2. STANDARDIZATION
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 3. WINDOWING (200ms Memory)
def create_windows(X, y, window_size=20):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

WINDOW_SIZE = 20
X_win, y_win = create_windows(X_scaled, y_raw, WINDOW_SIZE)

# Split into Train/Test
split = int(0.8 * len(X_win))
X_train, X_test = X_win[:split], X_win[split:]
y_train, y_test = y_win[:split], y_win[split:]

# 4. BUILD THE CNN-LSTM
model = Sequential([
    # CNN: Scan for gait patterns
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(WINDOW_SIZE, len(features))),
    MaxPooling1D(pool_size=2),
    
    # LSTM: Maintain the walking rhythm
    LSTM(64, return_sequences=False),
    Dropout(0.2),
    
    # Dense: Calculate the final motor angle
    Dense(32, activation='relu'),
    Dense(1) 
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 5. TRAIN
print("\n🚀 Training the AI on EPIC Lab physics...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 6. VISUALIZE
predictions = model.predict(X_test)

plt.figure(figsize=(12, 5))
plt.plot(y_test[:300], label='Actual Knee Angle (EPIC Ground Truth)', color='orange')
plt.plot(predictions[:300], label='AI Predicted Angle (ESP32 Command)', linestyle='--')
plt.title('Echo-Gait: Final Model Accuracy')
plt.ylabel('Degrees')
plt.legend()
plt.show()

# Save for conversion
model.save("epic_prosthetic_model.h5")