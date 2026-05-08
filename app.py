# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 21:25:52 2026

@author: hafsa
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# 1. Load data and train models immediately inside app.py
df = pd.read_csv('ai 2020.csv')
df_processed = df.drop(['UDI', 'Product ID', 'Type'], axis=1)
df_processed['Power'] = df_processed['Torque [Nm]'] * df_processed['Rotational speed [rpm]']
df_processed['Temp_Diff'] = df_processed['Process temperature [K]'] - df_processed['Air temperature [K]']

X = df_processed.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y = df_processed['Machine failure']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_scaled, y)

max_wear = df_processed['Tool wear [min]'].max()
df_processed['RUL'] = max_wear - df_processed['Tool wear [min]']
X_reg = df_processed.drop(['Machine failure', 'RUL', 'Tool wear [min]', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
y_reg = df_processed['RUL']
rul_model = RandomForestRegressor(random_state=42)
rul_model.fit(X_reg, y_reg)

# 2. Streamlit UI starts here
st.title("Machine Health Dashboard")

air_temp = st.sidebar.slider("Air Temp [K]", 295.0, 305.0, 298.0)
proc_temp = st.sidebar.slider("Process Temp [K]", 305.0, 315.0, 308.0)
rot_speed = st.sidebar.number_input("Rotational Speed [rpm]", 1000, 3000, 1500)
torque = st.sidebar.number_input("Torque [Nm]", 10.0, 100.0, 40.0)

# 5. Process data and make predictions when the button is clicked


if st.button("Predict"):
    # Calculate derived features
    power = torque * rot_speed
    temp_diff = proc_temp - air_temp
    
    # Prepare input for both models
    input_data_class = np.array([[air_temp, proc_temp, rot_speed, torque, power, temp_diff, 0]])
    input_data_reg = np.array([[air_temp, proc_temp, rot_speed, torque, power, temp_diff]])
    
    # Predict failure and RUL
    fail = model.predict(scaler.transform(input_data_class))
    rul = rul_model.predict(input_data_reg)
    
    # Get numeric value from RUL prediction
    rul_minutes = rul[0]
    
    # FORMATTING TIME: Convert minutes to "Hours and Minutes"
    if rul_minutes < 60:
        formatted_rul = f"{rul_minutes:.0f} minutes"
    else:
        hours = int(rul_minutes // 60)
        minutes = int(rul_minutes % 60)
        formatted_rul = f"{hours} hours {minutes} minutes"
        
    # DISPLAY RESULTS
    if fail[0] == 1:
        st.error("⚠️ STATUS: CRITICAL - Failure risk detected!")
    else:
        st.success("✅ STATUS: NORMAL - Machine is operating safely.")
        
    # Show the formatted time
    st.markdown(f"### Estimated Remaining Useful Life (RUL): **{formatted_rul}**")
