import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df= pd.read_csv("bitcoin_data.csv")
# print(df.head())

# print(df.info())

print(df.describe())

df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values("Timestamp")
df = df.ffill()

# Trend
plt.figure(figsize=(12,6))
plt.plot(df['Timestamp'], df['Close'])
plt.title("Bitcoin Price Trend")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid()
plt.show()

# Rolling Mean
df['MA30'] = df['Close'].rolling(30).mean()
df['MA90'] = df['Close'].rolling(90).mean()

plt.figure(figsize=(12,6))
plt.plot(df['Timestamp'], df['Close'], label='Price')
plt.plot(df['Timestamp'], df['MA30'], label='30d MA')
plt.plot(df['Timestamp'], df['MA90'], label='90d MA')
plt.title("Bitcoin Price with 30-Day and 90-Day Moving Averages")
plt.legend()
plt.grid()
plt.show()

# Volatility
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(30).std()

plt.figure(figsize=(12,6))
plt.plot(df['Timestamp'], df['Volatility'])
plt.title("30-day Volatility")
plt.grid()
plt.show()

df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Weekday'] = df['Timestamp'].dt.weekday

plt.figure(figsize=(10,5))
sns.boxplot(x='Month', y='Close', data=df)
plt.title("Monthly Seasonality")
plt.show()