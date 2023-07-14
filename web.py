import numpy as np
import pickle
import streamlit as st



loaded_model = pickle.load(open('trained_model.sav', 'rb'))

# Creating a function for prediction

def diabetes_prediction(input_data):

    # changing data a numpy array
    input_datarry = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_dataa = input_datarry.reshape(1, -1)

    prediction = loaded_model.predict(input_dataa)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else: 
        return 'This person is diabetic'
    
def main():

    # Giving a title
    st.title('Diabetes Prediction Web App')

    # getting input data from user
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Value')
    BloodPresure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Value')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of person')

    # Code for prediction
    diagnosis = ''

    # Creating a button for predictions
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPresure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)

if __name__ == '__main__':
    main()

