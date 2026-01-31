import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("bitcoin_data.csv")
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.sort_values("Timestamp")
df.set_index("Timestamp", inplace=True)
df = df.asfreq('D').ffill()

result = seasonal_decompose(df['Close'], model='additive', period=30)
result.plot()
plt.show()
