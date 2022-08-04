from datetime import date
import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px

# Site Config
st.set_page_config(

    page_title="Soligence IST Platform",
    page_icon=":shark:",
    layout="wide",
    menu_items= {
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

# Past and Present
START_DATE = '2017-01-01'
TODAY = date.today().strftime('%Y-%m-%d')

st.title(' Soligence Trading Platform ')

# Crypto List
currencies = ("BTC-USD","ETH-USD","USDT-USD","BNB-USD","XRP-USD", 
            "USDC-USD", "ADA-USD", "DOGE-USD", "SOL-USD", "BUSD-USD",
            "HEX-USD", "DAI-USD", "DOT-USD", "TRX-USD", "SHIB-USD", 
            "MATIC-USD", "AVAX-USD", "WBTC-USD", "LEO-USD", "LTC-USD")

selected_currency = st.selectbox("Select dataset for prediction", currencies)
st.text('Pricing data is updated frequently. Currency in USD')

# min: 1 max: 4
time_range = st.slider("Years of prediction:", 1, 4)
time_period = time_range * 365

# fetch selected coin dataset
def load_data(tickers):
    data = yf.download(tickers, START_DATE, TODAY)
    data.reset_index(inplace=True)

    return data

fetch_data = st.text('🏃🏽🏃🏽 Loading data 🏃🏽🏃🏽')

data = load_data(selected_currency)

fetch_data.text('Data loaded 🚀🚀')


sub_header = f'{selected_currency}: [{START_DATE} - {TODAY}] Dataset'
st.subheader(sub_header)
st.write(data)


# Visualization
def plot_dataset():
    try:

        # Scatter Graph
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name= 'Opened'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Adj Close'], name= 'Adj Close'))
    
        # Range Slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        ) 

        st.plotly_chart(fig)


        # Candle Stick
        fig.add_trace(go.Candlestick(x=data['Date'], open=data['Open'], 
                                            high=data['High'], low=data['Low'], 
                                            close=data['Close']))

        # Range Slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )                                    

        st.plotly_chart(fig)


        # Bar Chart
        fig = px.bar(data, x=data.Date, y='High')

        # Range Slider
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        st.plotly_chart(fig)
    except:
        st.error('Something went wrong somewhere 🤔🤔')
        print('An error occurred')

plot_dataset()



# Predict the Future
def predict_the_future():
    try:
        df_train_option = data[['Date', 'Close', 'Open', 'High', 'Low']]
        df_train_option = df_train_option.rename(columns={"Date": "ds", "Close": "y", 
                                                        "Open": "open", "High": "high", "Low": "low"})

        algo = Prophet()
        algo.fit(df_train_option)

        df_to_predict = algo.make_future_dataframe(periods=time_period)
        forecast = algo.predict(df_to_predict)

        sub_sub_header = f'Forecasted Dataset in [{time_period} Days] time'
        st.subheader(sub_sub_header)
        st.write(forecast)

        # Visualize forecasted Data
        fig1 = plot_plotly(algo, forecast)
        st.plotly_chart(fig1)

        st.subheader('Predictive Dataset Component')
        fig2 = algo.plot_components(forecast)
        st.write(fig2)
    except:
        st.error('Something went wrong somewhere 🤔🤔')
        print('An error occured')


predict_the_future()