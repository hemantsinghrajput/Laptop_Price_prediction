import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.title("Laptop Price Prediction 💻💻")
df=pickle.load(open('/app/laptop_price_prediction/Laptop/data.pkl', 'rb'))
model = pickle.load(open('/app/laptop_price_prediction/Laptop/laptop.pkl', 'rb'))

processor = st.selectbox("Select the Processor:- ", df["Processor"].unique())
ram = st.selectbox("Select the RAM:- ", df["Ram"].unique())
os = st.selectbox("Select the Operating Syatem:- ", df["OS"].unique())
Storage = st.selectbox("Select the Storage:- ", df["Storage"].unique())
st.write("Predict the Price of the Laptop of your choice❓")
butt = st.button("Predict ❗")

if butt:
    query = np.array([processor, ram, os, Storage])
    query = query.reshape(1, -1)
    p = model.predict(query)[0]
    result = np.exp(p)
    st.subheader("Your Predicted Prize is: ")
    st.subheader(":red[₹{}]".format(result.round()))


st.text("Developed by Hemant Singh Rajput")
