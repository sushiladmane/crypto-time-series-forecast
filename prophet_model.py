from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("bitcoin_data.csv")
df = df.tail(10000)
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df[['Timestamp','Close']]
df.columns = ['ds','y']
df = df.dropna()

model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True
)
model.fit(df)


future = model.make_future_dataframe(30)
forecast = model.predict(future)

model.plot(forecast)
plt.title("Bitcoin Price Forecast (Prophet)")
plt.show()

