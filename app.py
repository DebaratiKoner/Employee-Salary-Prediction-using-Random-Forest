import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Employee Salary Prediction using Random Forest", page_icon="💼", layout="centered")

st.markdown("""
    <style>
    /* App background and base text color */
    .stApp {
        background-color: black;
        color: white;
    }

    /* Sidebar background */
    section[data-testid="stSidebar"] {
        background-color: #111;
    }

    /* Main Page Title (h1) */
    h1, .stApp h1 {
        color: #a020f0; /* Purple */
        font-weight: bold;
    }

    /* Subheadings: Input Summary, Prediction Result */
    h2, h3, .stMarkdown h2, .stMarkdown h3 {
        color: #a020f0 !important; /* Purple */
    }

    /* Sidebar Header: Input Employee Details */
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #a020f0 !important;
    }

    /* Input Labels (age, workclass, etc.) */
    label, .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #a020f0 !important;
        font-weight: 500;
    }

    /* Prediction result text like "The predicted income is: <=50K" stays white */
    .stInfo, .stSuccess {
        color:  white !important;
        background-color:  #222 !important;
        border-left: 5px solid #4a00e0;
        padding: 1em;
    }
    .stInfo > div, .stSuccess > div {
        color: white !important;
}

    /* Buttons */
    .stButton > button {
        background-color: #4a00e0;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5em 1.2em;
        font-weight: bold;
        transition: background 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #8e2de2;
    }

    /* Input fields and dropdowns */
    .stTextInput > div > div > input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] {
        background-color: #222 !important;
        color: white !important;
        border-radius: 6px;
        border: 1px solid #444;
    }

    /* DataFrame style */
    .css-1y4p8pa {
        background-color: #000000;
        color: white;
    }

    /* --- Custom Additions for Slider and Prediction Text --- */

    /* Age Slider - Track and Fill */
    .stSlider > div > div > div[data-baseweb="slider"] > div:first-child { /* Slider track */
        background: #444 !important; /* Darker grey for the background track */
    }
    .stSlider > div > div > div[data-baseweb="slider"] > div:nth-child(2) { /* Filled part of the slider */
        background: #a020f0 !important; /* Purple for the filled part */
    }
    .stSlider .has-thumb > div > div[role="slider"] { /* Slider thumb/handle */
        background-color: #a020f0 !important; /* Purple for the slider thumb */
        border: 1px solid #a020f0 !important;
    }

    /* Age Slider - Thumb/Handle */
    .stSlider .has-thumb > div > div[role="slider"] {
        background-color: #a020f0 !important; /* Purple for the slider thumb */
        border: 1px solid #a020f0 !important;
    }

    /* Age Slider - Numbers (values) */
    .stSlider .st-bd, .stSlider .st-bg { /* These target the numerical values displayed on the slider */
        color: white !important;
    }

    /* Ensure .stInfo and .stSuccess also have white text if they contain the prediction */
    .stInfo > div, .stSuccess > div {
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)

# --- Load resources ---
@st.cache_resource
def load_resources():
    model = joblib.load("best_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    # Load a sample of the cleaned CSV to get options for selectboxes
    df = pd.read_csv("adult 3.csv")
    df['occupation'] = df['occupation'].replace({'?': 'Others'})
    df = df[~df['workclass'].isin(['Without-pay', 'Never-worked'])]
    df = df[~df['education'].isin(['5th-6th', '1st-4th', 'Preschool'])]
    return model, label_encoders, scaler, df

model, label_encoders, scaler, df_options = load_resources()

# --- UI ---
st.title("💼 Employee Salary Prediction using Random Forest")
st.markdown("""
Predict whether an employee earns *>50K* or *<=50K* based on their details.<br>
<b>Powered by Random Forest and SMOTE for balanced predictions.</b>
""", unsafe_allow_html=True)

st.sidebar.header("Input Employee Details")

# --- Input fields ---
age = st.sidebar.slider("Age", int(df_options['age'].min()), int(df_options['age'].max()), 30)
workclass = st.sidebar.selectbox("Workclass", sorted(df_options['workclass'].unique()))
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", int(df_options['fnlwgt'].min()), int(df_options['fnlwgt'].max()), 150000)
education = st.sidebar.selectbox("Education", sorted(df_options['education'].unique()))
educational_num = st.sidebar.selectbox("Educational Num", sorted(df_options['educational-num'].unique()))
marital_status = st.sidebar.selectbox("Marital Status", sorted(df_options['marital-status'].unique()))
occupation = st.sidebar.selectbox("Occupation", sorted(df_options['occupation'].unique()))
relationship = st.sidebar.selectbox("Relationship", sorted(df_options['relationship'].unique()))
race = st.sidebar.selectbox("Race", sorted(df_options['race'].unique()))
gender = st.sidebar.selectbox("Gender", sorted(df_options['gender'].unique()))
capital_gain = st.sidebar.number_input("Capital Gain", int(df_options['capital-gain'].min()), int(df_options['capital-gain'].max()), 0)
capital_loss = st.sidebar.number_input("Capital Loss", int(df_options['capital-loss'].min()), int(df_options['capital-loss'].max()), 0)
hours_per_week = st.sidebar.number_input("Hours per Week", int(df_options['hours-per-week'].min()), int(df_options['hours-per-week'].max()), 40)
native_country = st.sidebar.selectbox("Native Country", sorted(df_options['native-country'].unique()))

st.markdown("### Input Summary")
input_dict = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'education': education,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}
st.dataframe(pd.DataFrame([input_dict]))

# --- Prediction ---
if st.button("Predict Salary Class"):
    # Prepare input DataFrame
    input_df = pd.DataFrame([input_dict])

    # Label encode categorical columns (use hyphens, not underscores)
    categorical_cols = [
        'workclass', 'marital-status', 'occupation', 'relationship',
        'race', 'gender', 'native-country', 'education'
    ]
    for col in categorical_cols:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Scale numerical columns (use hyphens, not underscores)
    numerical_cols = [
        'age', 'fnlwgt', 'educational-num', 'capital-gain',
        'capital-loss', 'hours-per-week'
    ]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Ensure column order matches model training
    feature_order = [
        'age', 'workclass', 'fnlwgt', 'education', 'educational-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
    ]
    input_df = input_df[feature_order]

    # Predict
    pred = model.predict(input_df)[0]
    label_map = {0: "<=50K", 1: ">50K"}
    result = label_map.get(pred, pred)

    st.subheader("Prediction Result")
    if result == ">50K":
        st.markdown("🎉 The predicted income is: <span style='color:white'><b>*>50K*</b></span>", unsafe_allow_html=True)
    else:
        st.markdown(" 🎉The predicted income is: <span style='color:white'><b>*<=50K*</b></span>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<span style='color:#007BFF;'>© 2025 Employee Salary Prediction App.</span>", unsafe_allow_html=True)
st.markdown("<span style='color:#007BFF;'>Developed using Streamlit, Scikit-learn, and Random Forest.</span>", unsafe_allow_html=True)
