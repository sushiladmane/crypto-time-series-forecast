import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
df = pd.read_csv("bitcoin_data.csv")
df = df.tail(50000)
data = df[['Close']].values

# Scale
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# Create sequences
X, y = [], []
for i in range(60, len(scaled)):
    X.append(scaled[i-60:i])
    y.append(scaled[i])

X, y = np.array(X), np.array(y)

# Train test split
split = int(len(X)*0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(60,1)),
    Dropout(0.2),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train,epochs=30,batch_size=64,validation_split=0.1)

# Prediction
pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)
real = scaler.inverse_transform(y_test)

# Plot
plt.plot(real, label="Real")
plt.plot(pred, label="Predicted")
plt.legend()
plt.show()

