import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

#Header
st.set_page_config(page_title="Predict severity of the road accidents" ,page_icon="🚘", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('road_accident1.csv')
    return data

def preprocess_data(data):
    # Example preprocessing: handle missing values, encode categorical features, etc.
    # Convert non-numeric values in 'age_of_casualty' to NaN
    data['age_of_casualty'] = pd.to_numeric(data['age_of_casualty'], errors='coerce')
    
    # Drop rows with NaN values
    data = data.dropna()
    
    # Select relevant features
    X = data[['sex_of_casualty', 'age_of_casualty']]
    y = data['casualty_severity']
    
    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    return train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model
@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

data = load_data()
X_train, X_test, y_train, y_test = preprocess_data(data)
model = train_model(X_train, y_train)

# Title of the page
st.title(':blue[Road Accident Prediction]')

# Sidebar for navigation
st.sidebar.title(':blue[Navigation]')
page = st.sidebar.selectbox('Select a page:', ['Home', 'Data Visualization', 'Prediction'])

if page == 'Home':
    st.header(':red[Home]')
    st.write("Welcome to the Road Accident Prediction App. Use the navigation menu to explore the app.")
    st.image('r1.jpg', use_column_width=True)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = plt.imread(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)   

elif page == 'Data Visualization':
    st.header(':red[Data Visualization]')
    st.write("This page shows various visualizations of the road accident data.")
    
    # Display a bar plot for casualty severity distribution
    fig, ax = plt.subplots()
    sns.countplot(x='casualty_severity', data=data, ax=ax)
    st.pyplot(fig)

    # Display a violin plot for age of casualty by severity
    fig, ax = plt.subplots()
    sns.violinplot(x='casualty_severity', y='age_of_casualty', data=data, ax=ax)
    st.pyplot(fig)


elif page == 'Prediction':
    st.header(':red[Prediction]')
    st.write("This page allows you to input features and predict the severity of a road accident.")

    # Sidebar for user input
    st.sidebar.subheader('User Input Features')
    age_of_casualty = st.sidebar.slider('Age of Casualty', min_value=0, max_value=int(data['age_of_casualty'].max()), value=30)
    sex_of_casualty = st.sidebar.selectbox('Sex of Casualty', options=[1, 2], format_func=lambda x: 'Male' if x == 1 else 'Female')

    if st.sidebar.button('Predict'):
        # Preprocess the inputs appropriately
        input_data = np.array([[sex_of_casualty, age_of_casualty]])
        prediction = model.predict(input_data)
        st.write(f'The predicted casualty severity is: {prediction[0]}')

    # Show progress and status updates
    with st.spinner('Loading data and training model...'):
        accuracy = accuracy_score(y_test, model.predict(X_test))
        report = classification_report(y_test, model.predict(X_test))
        confusion = confusion_matrix(y_test, model.predict(X_test))
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(report)
    st.text("Confusion Matrix:")
    st.write(confusion)




