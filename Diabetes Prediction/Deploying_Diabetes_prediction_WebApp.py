# -*- coding: utf-8 -*-
"""
Created on Sat Jun  7 19:38:27 2025

@author: Priyanshu tiwari
"""

import numpy as np
import streamlit as st
import pickle

loaded_model = pickle.load(open('C:/Users/Priyanshu tiwari/Downloads/trained_model.sav', 'rb'))

def diabetes_prediction(input_data):
    
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    #Giving Title 
    
    st.title('Diabetes Prediction Web App')
    
    #Getting the input data from the user
    
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose Level')
    BloodPressure=st.text_input('BloodPressure Level')
    SkinThickness=st.text_input('Thickness of Skin')
    Insulin=st.text_input('Insulin Level')
    BMI=st.text_input('Value of BMI')
    DiabetesPedigreeFunction=st.text_input('Enter DiabetesPedigreeFunction')
    Age=st.text_input('Enter Age')
   
    #Code for Prediction
    
    diagnosis=''
    
    #Creating a button for diagnosis
    
    if st.button('Diabetes Prediction Test'):
        
        diagnosis=diabetes_prediction([ Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
        
    st.success(diagnosis)
    
    
    
    
if __name__ == '__main__':
    main()

