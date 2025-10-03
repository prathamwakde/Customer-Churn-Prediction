import streamlit as st
import joblib
import numpy as np

# Load the trained model and scaler
try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
except FileNotFoundError:
    st.error("Error: Model files ('scaler.pkl' or 'model.pkl') not found.")
    st.stop()

st.title("Churn Prediction App")
st.divider()
st.write("Please enter the value and hit the predict button for getting a prediction")
st.divider()

# Use a form to group inputs and prevent reruns on each character typed
with st.form("prediction_form"):
    age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
    tenure = st.number_input("Enter tenure", min_value=0, max_value=130, value=10)
    
    # CORRECTED: min_value must be less than max_value
    # Set a more appropriate default value within the new range
    monthly_charges = st.number_input("Enter monthly charges", min_value=10.0, max_value=150.0, value=70.0)
    
    gender = st.selectbox("Enter the Gender", ["Male", "Female"])
    st.divider()
    
    # A form_submit_button is required to submit the form
    predict_button = st.form_submit_button("Predict")

if predict_button:
    gender_selected = 1 if gender == "Male" else 0
    x = [age, tenure, monthly_charges, gender_selected]
    x1 = np.array(x)
    
    # The scaler expects a 2D array, so we wrap x1 in an outer list
    x_array = scaler.transform([x1])
    
    # Make the prediction
    prediction = model.predict(x_array)[0]
    
    # Convert prediction to 'Yes' or 'No' and display
    prediction_text = 'Yes' if prediction == 1 else 'No'
    
    if prediction == 1:
        st.error(f'Prediction: {prediction_text} (This customer is likely to churn)')
    else:
        st.success(f'Prediction: {prediction_text} (This customer is likely to stay)')

