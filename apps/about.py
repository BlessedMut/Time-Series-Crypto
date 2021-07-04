import streamlit as st

def app():
    
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