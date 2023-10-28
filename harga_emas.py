import pickle
import streamlit as st

model = pickle.load(open('Prediksi_harga_emas.sav', 'rb'))

st.title('prediksi Harga emas di china')

trade_date = st.number_input('Input Tahun Pembelian')
close = st.number_input('Input harga akhir')
open = st.number_input('Input harga awal')
high = st.number_input('Input harga tertinggi')
low = st.number_input('Input harga terendah')
change = st.number_input('Input perubahan harga')
vol = st.number_input('Input volume berat emas')
amount = st.number_input('Input jumlah emas')

predict = ''

if st.button('Estimasi Harga'):
    predict = model.predict(
        [[trade_date,close,open,high,low,change,vol,amount]]
    ) 
    st.write('prediksi harga emas dalam ponds : ', predict)
    st.write('Harga emas dalam yen : ', predict*6.4)
    st.write('Harga emas dalam IDR(Juta) : ', predict*14000)