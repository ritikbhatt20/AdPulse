import streamlit as st
import numpy as np
import pickle
import pandas as pd
from PIL import Image

# Loaded the pre-trained model and scaler
loaded_model = pickle.load(open('advertise_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl', 'rb'))

def preprocess_input_data(input_data, scaler):
    feature_names = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']

    # DataFrame with feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)

    input_df['Male'] = input_df['Male'].astype('bool')

    input_data_scaled = scaler.transform(input_df[feature_names])

    return input_data_scaled

# Set page title and icon
st.set_page_config(
    page_title="Ad Click Predictor",
    page_icon=":money_with_wings:"
)

st.title("Ad Click Predictor 🚀")

# Sidebar with input controls
st.sidebar.header("User Inputs")
age = st.sidebar.slider("Age", 18, 70, 30)
area_income = st.sidebar.slider("Area Income", 30000, 80000, 50000)
daily_internet_usage = st.sidebar.slider("Daily Internet Usage", 50, 250, 150)
daily_time_spent = st.sidebar.slider("Daily Time Spent on Site", 20, 120, 60)
is_male = st.sidebar.checkbox("Male")

input_data = {
    'Age': age,
    'Area Income': area_income,
    'Daily Internet Usage': daily_internet_usage,
    'Daily Time Spent on Site': daily_time_spent,
    'Male': is_male,
}

st.subheader("Input Data:")
st.write(input_data)

input_data_processed = preprocess_input_data(input_data, loaded_scaler)
st.subheader("Processed Input Data:")
st.write(input_data_processed)

prediction = loaded_model.predict(input_data_processed)

st.subheader("Prediction:")
if prediction[0] == 0:
    st.write('The user is not likely to click on the ad. 😕')
else:
    st.write('The user is likely to click on the ad! 🎉')


