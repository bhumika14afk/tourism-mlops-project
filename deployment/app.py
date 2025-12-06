
'''
1. Download the model from the Model Hub.
2. Load the model.
3. Streamlit UI for tourism Prediction.
4. Collect user input.
5. Prepare input data.
6. Create Predict button on the user interface.

'''
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
import os

# Page configuration
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="wide"
)

# Download model from Hugging Face
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="bhumikam14/tourism-package-model",
            filename="tourism_model.pkl",
            repo_type="model"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

# Title
st.title("✈️ Tourism Package Prediction System")
st.markdown("### Predict whether a customer will purchase the Wellness Tourism Package")

if model is None:
    st.error("Failed to load the model. Please check the Hugging Face repository.")
else:
    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Personal Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
        occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
        monthly_income = st.number_input("Monthly Income (₹)", min_value=0, value=20000, step=1000)
        
    with col2:
        st.subheader("🏨 Travel Preferences")
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        number_of_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
        number_of_children = st.number_input("Number of Children Visiting (below 5 years)", min_value=0, max_value=5, value=0)
        preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
        number_of_trips = st.number_input("Average Number of Trips Per Year", min_value=0.0, max_value=20.0, value=1.0, step=0.5)
        passport = st.selectbox("Have Passport?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        own_car = st.selectbox("Own Car?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("💼 Interaction Details")
    col3, col4 = st.columns(2)

    with col3:
        type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
        product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
        duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=120.0, value=15.0, step=1.0)

    with col4:
        pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
        number_of_followups = st.number_input("Number of Follow-ups", min_value=0.0, max_value=10.0, value=2.0, step=1.0)

    # Prediction button
    if st.button("🔮 Predict Purchase Probability", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = pd.DataFrame({
            'Age': [age],
            'TypeofContact': [type_of_contact],
            'CityTier': [city_tier],
            'DurationOfPitch': [duration_of_pitch],
            'Occupation': [occupation],
            'Gender': [gender],
            'NumberOfPersonVisiting': [number_of_persons],
            'NumberOfFollowups': [number_of_followups],
            'ProductPitched': [product_pitched],
            'PreferredPropertyStar': [preferred_property_star],
            'MaritalStatus': [marital_status],
            'NumberOfTrips': [number_of_trips],
            'Passport': [passport],
            'PitchSatisfactionScore': [pitch_satisfaction_score],
            'OwnCar': [own_car],
            'NumberOfChildrenVisiting': [number_of_children],
            'Designation': [designation],
            'MonthlyIncome': [monthly_income]
        })
        
        try:
            # Make prediction
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            if prediction == 1:
                st.success("✅ **Customer is LIKELY to purchase the package!**")
            else:
                st.error("❌ **Customer is UNLIKELY to purchase the package**")
            
            # Show probability breakdown
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("No Purchase Probability", f"{prediction_proba[0]:.1%}")
            with col_b:
                st.metric("Purchase Probability", f"{prediction_proba[1]:.1%}")
            
            # Progress bar
            st.progress(prediction_proba[1])
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.markdown("**Visit with Us** - Tourism Package Prediction System")
