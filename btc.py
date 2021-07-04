import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import model_from_json
import os.path
from tensorflow import keras
import streamlit as st
import numpy as np
from app import days

# db_connection = sql.connect(host='localhost', database='timeseries', user='root', password='')

# test_data = pd.read_sql('SELECT * FROM btc_usdt', con=db_connection)

crypto_currency = 'BTC'
base_currency = 'USD'

prediction_days = 60

def get_train_data():
  n_days = days
  data = pd.read_csv('./data/btc_usdt.csv', header=None)
  data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
  data['Date'] = pd.to_datetime(data['Date'])
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))

  x_train, y_train = [], []

  if n_days > 1:
    for x in range(prediction_days, len(scaled_data)-n_days):
      x_train.append(scaled_data[x-prediction_days:x, 0])
      y_train.append(scaled_data[x+n_days, 0])
  else:
    for x in range(prediction_days, len(scaled_data)):
      x_train.append(scaled_data[x-prediction_days:x, 0])
      y_train.append(scaled_data[x, 0])

  x_train, y_train = np.array(x_train), np.array(y_train)
  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

  return data, scaler, x_train, y_train

def get_test_data():
  data = get_train_data()[0]
  scaler = MinMaxScaler(feature_range=(0,1))
  test_start = "2018-01-01"
  test_data = pd.read_csv('./data/btc_usdt.csv', header=None)
  test_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
  test_data['Date'] = pd.to_datetime(test_data['Date'])
  test_data = test_data.loc[test_data['Date']>=test_start]
  actual_values = test_data['Close'].values
  scaler = MinMaxScaler(feature_range=(0,1))
  scaled_data = scaler.fit_transform(actual_values.reshape(-1,1))
  total_data = pd.concat((data['Close'], test_data['Close']), axis=0)
  model_inputs = total_data[len(total_data)-len(test_data)-prediction_days:].values
  model_inputs = model_inputs.reshape(-1,1)
  model_inputs = scaler.fit_transform(model_inputs)

  x_test, y_test = [], []

  for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])
    y_test.append(model_inputs[x, 0])

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
