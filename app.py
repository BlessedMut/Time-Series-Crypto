import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from btc import *

model = load_trained_model()
tomorrow = btc_pred()

# test_data = pd.read_csv('./data/btc_usdt.csv', header=None)
# test_data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
# test_data['Date'] = pd.to_datetime(test_data['Date'])

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
        st.write('Rufaro Nyandoro - R182565F')
        st.write('rufarohazelnyandoro@gmail.com')
    with m2:
        st.write("|")
    with l:
        st.write('Lyka  - R182565F')
        st.write('lyka@gmail.com')


def updates():
    st.title("Market Updates")

    d1, d2 = st.beta_columns([3,6])

    with d1:
        days = st.slider("Predict how many days in the future", min_value=1, max_value=15, step=1, value=1)

    with d2:
        st.write("")
        st.write("")
        st.markdown(f"The estimated price for {days} day(s) in the future is __${tomorrow:.2f}")

    plot_data = test_data[['Date','Close']]

    st.line_chart(plot_data.rename(columns={'Date':'index'}).set_index('index'))
    

def recommendations():
    st.title("Market Recommendations")

    col1, col2 = st.beta_columns([8,3])

    with col1:
        st.dataframe(test_data)

    with col2:
        crypto = st.selectbox(
                'Select Crypto-Currency',
                ['BTC', 'ETH', 'PI', 'DGE']
            )

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

