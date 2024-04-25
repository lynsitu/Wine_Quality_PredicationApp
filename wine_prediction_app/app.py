#import libraries
import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict

#app heading
st.write("""
# Wine Quality Prediction App
This app predicts the ***Wine Quality*** type!
""")
#creating sidebar for user input features
st.sidebar.header('User Input Parameters')

fixed_acidity = st.sidebar.slider('fixed acidity', 3.8, 15.9, 7.21)
volatile_acidity = st.sidebar.slider('volatile acidity', 0.08,1.58 , 0.34)
citric_acid = st.sidebar.slider('citric acid', 0.0, 1.66, 0.31)
residual_sugar = st.sidebar.slider('residual sugar', 0.6, 65.8 ,5.44)
chlorides = st.sidebar.slider('chlorides', 0.01,0.61 , 0.06)
free_sulfur_dioxide = st.sidebar.slider('free sulfur dioxide', 1.0,289.0 , 30.53)
total_sulfur_dioxide = st.sidebar.slider('total sulfur dioxide', 6.0, 440.0, 115.74)
density= st.sidebar.slider('density', 0.99007,1.00369 , 0.9947)
pH = st.sidebar.slider('pH', 2.72,4.01 , 3.2)
sulphates = st.sidebar.slider('sulphates', 0.22, 2.0, 0.53)
alcohol=st.sidebar.slider('alcohol', 8.0,14.9, 10.5)

features = pd.DataFrame({'fixed acidity': fixed_acidity,
                        'volatile acidity': volatile_acidity,
                        'citric acid': citric_acid,
                        'residual sugar': residual_sugar,
                        'chlorides': chlorides,
                        'free sulfur dioxide': free_sulfur_dioxide,
                        'total sulfur dioxide': total_sulfur_dioxide,
                        'density': density,
                        'pH': pH,
                        'sulphates': sulphates,
                        'alcohol': alcohol}, index=[0])

st.subheader('User Input parameters')
st.write(features)


st.subheader('Wine quality labels - There are 6 classes of wine quality, from 3 to 8')
st.write(pd.DataFrame({
   'wine quality': [3, 4, 5, 6, 7, 8 ]}, index=range(1,7)))


st.subheader('Prediction - Wine Quality Class')  
result = predict(np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]).reshape(1, -1))
st.text(result[0])