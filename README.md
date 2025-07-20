# Employee-Salary-Prediction-using-Random-Forest

## Overview
This project is a web application built with Streamlit that predicts whether an employee earns **>50K** or **<=50K** based on their personal and professional details. The prediction is powered by a Random Forest model trained on the "adult 3.csv" dataset, with SMOTE applied for class balancing.

## Features
- Interactive UI for entering employee details
- Real-time salary class prediction
- Uses Random Forest and SMOTE for robust, balanced results
- Clean, professional interface with dark theme

## Technologies Used
- Python
- Streamlit
- scikit-learn
- pandas
- numpy
- joblib

## Files
- `app.py`: Main Streamlit application
- `best_model.pkl`: Trained Random Forest model
- `label_encoders.pkl`: Label encoders for categorical features
- `scaler.pkl`: MinMaxScaler for numerical features
- `adult 3.csv`: Cleaned dataset for selectbox options

## How to Run

1. **Install dependencies**  
   ```
   pip install streamlit pandas scikit-learn numpy joblib
   ```

2. **Ensure all required files are present**  
   - `app.py`
   - `best_model.pkl`
   - `label_encoders.pkl`
   - `scaler.pkl`
   - `adult 3.csv`

3. **Start the Streamlit app**  
   ```
   streamlit run app.py
   ```

4. **Use the sidebar to input employee details and get predictions.**

## Input Fields
- Age
- Workclass
- Final Weight (fnlwgt)
- Education
- Educational Num
- Marital Status
- Occupation
- Relationship
- Race
- Gender
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

## Output
- Predicted salary class: **>50K** or **<=50K**

## License & Policies
© 2025 Employee Salary Prediction App.  
Developed using Streamlit, Scikit-learn, and Random Forest.
