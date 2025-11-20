import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model
import sklearn.compose._column_transformer
class _RemainderColsList(list):
    pass
sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList


pipe = joblib.load('car_model.pkl')

st.title("🚗Car Price Predictor") 
st.markdown("Trained on 49k rows. RMS=726.65  ")
#users input
cat_cols = ['Manufacturer', 'Model', 'Fuel type']
manu   = st.selectbox('Manufacturer', sorted(pipe.named_steps['prep']
                .named_transformers_['cat'].categories_[0]))
model  = st.selectbox('Model', sorted(pipe.named_steps['prep']
                .named_transformers_['cat'].categories_[1]))
fuel   = st.selectbox('Fuel type', sorted(pipe.named_steps['prep']
                .named_transformers_['cat'].categories_[2]))

eng    = st.number_input('Engine size (L)', min_value=.5, max_value=6.0, value=1.6, step=.1)
year   = st.number_input('Year of manufacture', 1980, 2025, 2015)
mileage= st.number_input('Mileage', 0, 500_000, 60_000, step=1000)

# --- prediction ---
if st.button('Predict'):
    X_user = pd.DataFrame([{'Manufacturer': manu,
                            'Model': model,
                            'Fuel type': fuel,
                            'Engine size': eng,
                            'Year of manufacture': year,
                            'Mileage': mileage}])
    price = pipe.predict(X_user)[0]
    st.success(f'Estimated price:  **£{price:,.0f}**')