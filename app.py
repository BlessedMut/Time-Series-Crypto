import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json
import os.path
from tensorflow import keras
import streamlit as st
import numpy as np
import yfinance as yf
from yahoofinancials import YahooFinancials
import datetime as dt

# db_connection = sql.connect(host='localhost', database='timeseries', user='root', password='')

# test_data = pd.read_sql('SELECT * FROM btc_usdt', con=db_connection)

crypto_currency = 'BTC'
base_currency = 'USD'
global days
prediction_days = 60

start_date =  dt.datetime(2014,9,17)
end_date = dt.datetime.now()

data = yf.download(f'{crypto_currency}-{base_currency}', 
                      start=start_date, 
                      end=end_date, 
                      progress=False,)

def get_train_data():
#   data = pd.read_csv('./data/btc_usdt.csv', header=None)
#   data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
#   data['Date'] = pd.to_datetime(data['Date'])
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

  x_train, y_train = [], []

  for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

  return data, scaler, x_train, y_train

def get_test_data():
  data = get_train_data()[0]
  scaler = MinMaxScaler(feature_range=(0,1))
  test_start = "2018-01-01"
#   test_data = pd.read_csv('./data/btc_usdt.csv', header=None)
#   test_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
#   test_data['Date'] = pd.to_datetime(test_data['Date'])
  test_data = data.loc[data.index>=test_start]
  test_start = "2018-01-01"
  test_end = dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d")
  test_data = yf.download(f'{crypto_currency}-{base_currency}', 
                      start=test_start, 
                      end=test_end, 
                      progress=False,)
  actual_values = test_data['Close'].values
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(actual_values.reshape(-1,1))
  total_data = pd.concat((data['Close'], test_data['Close']), axis=0)
  model_inputs = total_data[len(total_data)-len(test_data)-prediction_days:].values
  model_inputs = model_inputs.reshape(-1,1)
  model_inputs = scaler.fit_transform(model_inputs)

  x_test, y_test = [], []

  if days <= 1:
    for x in range(prediction_days, len(model_inputs)):
      x_test.append(model_inputs[x-prediction_days:x, 0])
      y_test.append(model_inputs[x, 0])
  else:
    for x in range(prediction_days, len(model_inputs)-30):
      x_test.append(model_inputs[x-prediction_days:x, 0])
      y_test.append(model_inputs[x+30, 0])
    
  x_test, y_test = np.array(x_test), np.array(y_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

  pred_dates = total_data.index.values[-(len(x_test)):]
  return x_test, pred_dates, actual_values, model_inputs, x_test, y_test

def load_trained_model():
  with open('./trained_models/btc.json', 'r') as json_file:
    model = model_from_json(json_file.read())

    # load weights into new model
    model.load_weights("./trained_models/btc_model.h5")
  return model

model = load_trained_model()

def btc_pred():
  model_inputs = get_test_data()[3]
  scaler = get_train_data()[1]
  x_test = get_test_data()[4]
  r_data = [model_inputs[len(model_inputs)+ 1 - prediction_days:len(model_inputs)+1,0]]
  r_data = np.array(r_data)
  r_data = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))

  pred = model.predict(r_data)
  pred = scaler.inverse_transform(pred)
  return pred[-1][0]


test_data = pd.read_csv('./data/btc_usdt.csv', header=None)
test_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
test_data['Date'] = pd.to_datetime(test_data['Date'])

def about():
    st.title('Time Series Predictions for Crypto-Currencies')

    c1, c2, c3 = st.beta_columns([4,2,4])
    with c2:
        st.subheader('Done by:')

    b,m1,r, m2, l = st.beta_columns([2,1,2,1,2])
    with b:
        st.write('Blessed Mutengwa - R182565F')
        st.write('blessedmutengwa@gmail.com')
    with m1:
        st.write("|")
    with r:
        st.write('Rufaro Nyandoro - R1710732')
        st.write('rufarohazelnyandoro@gmail.com')
    with m2:
        st.write("|")
    with l:
        st.write('Laika Ali  - R178492M')
        st.write('laikaali@gmail.com')


def updates():
    global days
    st.title("Market Updates")

    d1, d2 = st.beta_columns([3,6])

    with d1:
        days = st.slider("Predict how many days in the future", min_value=1, max_value=15, step=1, value=1)
        get_train_data()
        tomorrow = btc_pred()


    with d2:
        st.write("")
        st.write("")
        pred = st.empty()
        pred.markdown(f"The estimated price for {days} day(s) in the future is __${tomorrow:.2f}__")

    plot_data = test_data[['Date','Close']]

    st.line_chart(plot_data.rename(columns={'Date':'index'}).set_index('index'))
    

def recommendations():
    st.title("Market Recommendations")
    
    st.subheader('Historical Price Data')
    
    col1, col2, col3 = st.beta_columns([1,10,1])

    crypto = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'ADA-USD', 'DOGE-USD', 'THETA-USD', 'LTC-USD', 'VET-USD', 'TRX-USD']

    yahoo_financials = YahooFinancials(crypto)

    d = yahoo_financials.get_historical_price_data(start_date='2021-06-29', 
                                                      end_date=dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d"), 
                                                      time_interval='weekly')
    prices_df = pd.DataFrame({
    a: {x['formatted_date']: x['adjclose'] for x in d[a]['prices']} for a in crypto
    })
              
    with col2:
      st.dataframe(prices_df)
    
    crypto = ['BTC-USD', 'ETH-USD', 'DOGE-USD']

    yahoo_financials_crypto = YahooFinancials(crypto)
    summary = yahoo_financials_crypto.get_summary_data()
    
    st.markdown('Comparing __BTC-USD__ to _ETH-USD_ & _DOGE-USD_')
    table1, table2, table3 = st.beta_columns(3)
    
    df = ['df1', 'df2', 'df3']

    for i, (k, v) in enumerate(summary.items()):
      df[i] = (pd.Series(v).to_frame(str(k)))
     
    with table1:
      tb1 = st.empty()
      tb1.dataframe(df[0])
    with table2:
      tb2 = st.empty()
      tb2.dataframe(df[1])
    with table3:
      tb2 = st.empty()
      tb2.dataframe(df[2])

    
def main():
    app = st.sidebar.selectbox(
                'Navigation',
                ['Market Updates', 'Market Recommendations', 'About Project'])

    

    if app == 'Market Updates':
        updates()
    elif app == 'Market Recommendations':
        recommendations()
    elif app == 'About Project':
        about()

if __name__ == '__main__':
    main()

