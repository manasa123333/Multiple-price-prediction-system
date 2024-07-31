# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 08:14:05 2024

@author: sanath
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import pandas as pd
import pyarrow
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Print versions to ensure everything is correctly installed
print(f"pyarrow version: {pyarrow.__version__}")
print(f"xgboost version: {xgb.__version__}")



house_model = pickle.load(open('C:/Users/sanath/Desktop/multiple disease prediction system/saved_models/HousePricePrediction.pkl','rb'))
car_model = pickle.load(open('C:/Users/sanath/Desktop/multiple disease prediction system/saved_models/CarPricePrediction.pkl','rb'))
bike_model = pickle.load(open('C:/Users/sanath/Desktop/multiple disease prediction system/saved_models/bikePrediction_model.pkl','rb'))



with st.sidebar:
    
    selected = option_menu('Multiple Price Prediction System', 
                           ['House Price Prediction',
                            'Car Price Prediction',
                            'Bike Price Prediction'],
                           
                           icons = ['house','car-front-fill','bicycle'],
                           default_index=0)
    

if selected == 'House Price Prediction':
    st.title('House Price Prediction')
    
    #input
    medinc = st.text_input('The median income of the block group (in tens of thousands of dollars)')
    HouseAge = st.text_input('The median age of the houses in the block group (in years)')
    AveRooms = st.text_input('The average number of rooms per household in the block group')
    AveBedrms = st.text_input(' the average number of bedrooms per household in a given block group.')
    Population = st.text_input('The total number of people living in a block group.')
    AveOccup = st.text_input('Average number of people per household')
    Latitude = st.text_input('Latitude of the block group')
    Longitude = st.text_input('Longitude of the block group')
    
    #output
    
    # creating a button for prediction
    
    if st.button('predict house price'):
    # Convert input data to floats
      input_data = np.array([[float(medinc), float(HouseAge), float(AveRooms), float(AveBedrms), 
                            float(Population), float(AveOccup), float(Latitude), float(Longitude)]])
      price_prediction = house_model.predict(input_data)
    
      st.success(f'Predicted house price: ${price_prediction[0]:,.2f}')

        


elif selected == 'Car Price Prediction':
    st.title('Car Price Prediction')
    
    name = st.text_input('car name')
    company = st.text_input('Company name')
    year = st.text_input('year')
    kms_driven = st.text_input('kilometers driven')
    fuel_type = st.text_input('Fuel Type')
    
    if st.button('Predict Car Price'):
        # Handle categorical variables with LabelEncoder
        label_encoder = LabelEncoder()
        name_encoded = label_encoder.fit_transform([name])[0]
        company_encoded = label_encoder.fit_transform([company])[0]
        fuel_type_encoded = label_encoder.fit_transform([fuel_type])[0]
        
        # Create input array
        inputs = np.array([[name_encoded, company_encoded, int(year), float(kms_driven), fuel_type_encoded]])
        
        # Scale the input
        scaler = StandardScaler()
        scaled_car = scaler.fit_transform(inputs)
        
        # Predict car price
        car_price_prediction = car_model.predict(scaled_car)
        st.success(f'Predicted car price: ${car_price_prediction[0]:,.2f}')

elif selected == 'Bike Price Prediction':
    st.title('Bike Price Prediction')
    
    bike_name = st.text_input('Name of the bike')
    kms_driven = st.text_input('Kilometers driven')
    owner = st.text_input('Owner')
    age = st.text_input('Age of the bike')
    power = st.text_input('Power of the bike')
    brand = st.text_input('Bike brand')
    
    if st.button('Predict Bike Price'):
        # Handle categorical variables with LabelEncoder
        label_encoder = LabelEncoder()
        bike_name_encoded = label_encoder.fit_transform([bike_name])[0]
        owner_encoded = label_encoder.fit_transform([owner])[0]
        brand_encoded = label_encoder.fit_transform([brand])[0]
        
        # Create input array
        inputs = np.array([[bike_name_encoded, float(kms_driven), owner_encoded, int(age), float(power), brand_encoded]])
        
        # Scale the input
        scaler = StandardScaler()
        scaled_bike = scaler.fit_transform(inputs)
        
        # Predict bike price
        bike_price_prediction = bike_model.predict(scaled_bike)
        st.success(f'Predicted bike price: ${bike_price_prediction[0]:,.2f}')
    