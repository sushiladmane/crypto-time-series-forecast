from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bitcoin_data.csv")
df = df.tail(5000)

df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values('Timestamp')
df.set_index('Timestamp', inplace=True)
df = df.asfreq('D').ffill() 

model = ARIMA(df['Close'], order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(30)

plt.figure(figsize=(10,5))
plt.plot(df['Close'], label='Actual')
plt.plot(range(len(df), len(df)+30), forecast, label='Forecast')
plt.legend()
plt.title("ARIMA Forecast (30 Days)")
plt.show()

