import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import load_model

# Load the models and scaler
xgb_model = joblib.load('models/xgb_model.joblib')
scaler = joblib.load('models/scaler.joblib')

# Load the model
st.write("Loading model...")
# xgb_model = load_model('models/ann_model.h5')

st.title('Activity Classification')
st.write('Input the acceleration values to classify the activity.')

x_input = st.number_input('X Acceleration')
y_input = st.number_input('Y Acceleration')
z_input = st.number_input('Z Acceleration')

if st.button('Classify'):
    input_data = np.array([[x_input, y_input, z_input]])
    input_data = scaler.transform(input_data)
    prediction = xgb_model.predict(input_data)
    activity_map = {
        0: 'Working at Computer',
        1: 'Standing Up, Walking and Going up/down stairs',
        2: 'Standing',
        3: 'Walking',
        4: 'Going Up/Down Stairs',
        5: 'Walking and Talking with Someone',
        6: 'Talking while Standing'
    }
    predicted_activity = activity_map.get(prediction[0], 'Unknown')
    st.write(f'Predicted Activity: {predicted_activity}')
