import datetime as dt
import mysql.connector as sql
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials
from sqlalchemy import create_engine

crypto_currency = 'BTC'
base_currency = 'USD'

start_date = "2014-09-17"
end_date = dt.datetime.strftime(dt.datetime.now(), "%Y-%m-%d")

data = yf.download(f'{crypto_currency}-{base_currency}', 
                      start=start_date, 
                      end=end_date, 
                      progress=False,
)
print(data.head())


engine = create_engine("mysql+pymysql://{user}:{pw}@{host}/{db}"
                .format(host='localhost', db='timeseries', user='root', pw=''))

data.to_sql(con=engine, name='btc_usdt', if_exists='replace')

