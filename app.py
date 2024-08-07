import streamlit as st
import pandas as pd
import joblib

# Load the model and scaler
model = joblib.load('lung_cancer_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to get user input
def get_user_input():
    gender = st.selectbox('Gender', ['Male', 'Female'])
    age = st.slider('Age', 0, 100, 50)
    smoking = st.slider('Smoking', 1, 2, 1)
    yellow_fingers = st.slider('Yellow Fingers', 1, 2, 1)
    anxiety = st.slider('Anxiety', 1, 2, 1)
    peer_pressure = st.slider('Peer Pressure', 1, 2, 1)
    chronic_disease = st.slider('Chronic Disease', 1, 2, 1)
    fatigue = st.slider('Fatigue', 1, 2, 1)
    allergy = st.slider('Allergy', 1, 2, 1)
    wheezing = st.slider('Wheezing', 1, 2, 1)
    alcohol_consuming = st.slider('Alcohol Consuming', 1, 2, 1)
    coughing = st.slider('Coughing', 1, 2, 1)
    shortness_of_breath = st.slider('Shortness of Breath', 1, 2, 1)
    swallowing_difficulty = st.slider('Swallowing Difficulty', 1, 2, 1)
    chest_pain = st.slider('Chest Pain', 1, 2, 1)

    # Convert gender to numerical value
    gender = 0 if gender == 'Male' else 1

    # Create a data frame
    user_data = {
        'GENDER': gender,
        'AGE': age,
        'SMOKING': smoking,
        'YELLOW_FINGERS': yellow_fingers,
        'ANXIETY': anxiety,
        'PEER_PRESSURE': peer_pressure,
        'CHRONIC DISEASE': chronic_disease,
        'FATIGUE': fatigue,
        'ALLERGY': allergy,
        'WHEEZING': wheezing,
        'ALCOHOL CONSUMING': alcohol_consuming,
        'COUGHING': coughing,
        'SHORTNESS OF BREATH': shortness_of_breath,
        'SWALLOWING DIFFICULTY': swallowing_difficulty,
        'CHEST PAIN': chest_pain
    }

    features = pd.DataFrame(user_data, index=[0])
    # Strip any leading or trailing spaces from column names
    features.columns = features.columns.str.strip()
    return features

# Streamlit UI
st.title('Lung Cancer Prediction')
st.write('By Olorunfemi Oluwadarasimi Samuel 20/47cs/01125')

st.write('Please input the following details:')

# Get user input
user_input = get_user_input()

# Scale the user input
scaled_input = scaler.transform(user_input)

# Predict
prediction = model.predict(scaled_input)
prediction_proba = model.predict_proba(scaled_input)

st.subheader('Prediction')
if prediction[0] == 0:
    st.write('No Lung Cancer')
else:
    st.write('Lung Cancer')

st.subheader('Prediction Probability')
st.write(prediction_proba)
