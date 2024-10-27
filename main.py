import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# Load data
def load_data():
    data = pd.read_csv('train_.csv')
    return data

# Preprocess data
def preprocess_data(data, is_training=True):
    if 'Loan_ID' in data.columns:
        data = data.drop('Loan_ID', axis=1)
    
    imputer = SimpleImputer(strategy='mean')
    data['LoanAmount'] = imputer.fit_transform(data[['LoanAmount']])
    data['Loan_Amount_Term'] = imputer.fit_transform(data[['Loan_Amount_Term']])
    data['Credit_History'] = imputer.fit_transform(data[['Credit_History']])
    
    for column in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        if column in data.columns:
            data[column].fillna(data[column].mode()[0], inplace=True)
    
    if 'Dependents' in data.columns:
        data['Dependents'] = data['Dependents'].replace('3+', 3).astype(int)
    
    label_enc = LabelEncoder()
    for column in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
        if column in data.columns:
            data[column] = label_enc.fit_transform(data[column])
    
    if is_training and 'Loan_Status' in data.columns:
        data['Loan_Status'] = label_enc.fit_transform(data['Loan_Status'])
    
    return data

# Train model
def train_model(data):
    X = data.drop('Loan_Status', axis=1)
    y = data['Loan_Status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, accuracy

# Streamlit UI
st.title('Loan Price Prediction')

# Load and preprocess data
data = load_data()
preprocessed_data = preprocess_data(data)

# Train the model
model, accuracy = train_model(preprocessed_data)

st.write('Model Accuracy: {:.2f}%'.format(accuracy * 100))

# Session state management with dropdown menus
st.header('Predict Loan Approval')

# Gender dropdown menu
gender_options = ['Male', 'Female']
st.session_state['Gender'] = st.selectbox('Gender', gender_options)

# Married dropdown menu
married_options = ['Yes', 'No']
st.session_state['Married'] = st.selectbox('Married', married_options)

# Education dropdown menu
education_options = ['Graduate', 'Not Graduate']
st.session_state['Education'] = st.selectbox('Education', education_options)

# Self Employed dropdown menu
self_employed_options = ['Yes', 'No']
st.session_state['Self_Employed'] = st.selectbox('Self Employed', self_employed_options)

# Property Area dropdown menu
property_area_options = ['Urban', 'Rural', 'Semiurban']
st.session_state['Property_Area'] = st.selectbox('Property Area', property_area_options)

# User input for other fields
dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])
applicant_income = st.number_input('Applicant Income', min_value=0)
coapplicant_income = st.number_input('Coapplicant Income', min_value=0)
loan_amount = st.number_input('Loan Amount', min_value=0)
loan_amount_term = st.number_input('Loan Amount Term (in months)', min_value=12, max_value=480)
credit_history = st.selectbox('Credit History', ['0', '1'])

# Create a prediction based on user input
user_input = pd.DataFrame({
    'Gender': [st.session_state['Gender']],
    'Married': [st.session_state['Married']],
    'Dependents': [dependents],
    'Education': [st.session_state['Education']],
    'Self_Employed': [st.session_state['Self_Employed']],
    'ApplicantIncome': [applicant_income],
    'CoapplicantIncome': [coapplicant_income],
    'LoanAmount': [loan_amount],
    'Loan_Amount_Term': [loan_amount_term],
    'Credit_History': [credit_history],
    'Property_Area': [st.session_state['Property_Area']]
})

# Preprocess user input
user_input = preprocess_data(user_input, is_training=False)

# Predict loan status
prediction = model.predict(user_input)[0]

# Display loan prediction with better formatting
if prediction == 1:
    st.success('Loan Prediction: Approved')
else:
    st.error('Loan Prediction: Not Approved')

