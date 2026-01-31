import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

st.title("Interactive Crypto Forecast Dashboard")

@st.cache_data
def load_data():
    df = pd.read_parquet("btc.parquet")
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
    df = df.sort_values("Timestamp")
    df.set_index("Timestamp", inplace=True)
    return df.tail(5000)

df = load_data()

# Sidebar controls
start_date = st.sidebar.date_input("Start Date", df.index.min())
end_date = st.sidebar.date_input("End Date", df.index.max())
model_choice = st.sidebar.selectbox("Model", ["ARIMA", "Prophet"])
n_days = st.sidebar.slider("Forecast Days", 7, 90, 30)

df = df.loc[start_date:end_date]

st.subheader("Historical Price")
st.line_chart(df['Close'])

st.subheader("Forecast")

if model_choice == "ARIMA":
    model = ARIMA(df['Close'], order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(n_days)
    future_dates = pd.date_range(df.index[-1], periods=n_days+1, freq='D')[1:]

    plt.plot(df['Close'], label="Actual")
    plt.plot(future_dates, forecast, label="Forecast")
    plt.legend()
    st.pyplot(plt)

else:
    prophet_df = df.reset_index()[['Timestamp','Close']]
    prophet_df.columns = ['ds','y']

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(n_days)
    forecast = model.predict(future)

    fig = model.plot(forecast)
    st.pyplot(fig)


