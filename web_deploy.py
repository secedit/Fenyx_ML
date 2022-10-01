#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 07:03:58 2022

@author: demir
"""

#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image



#load the model from disk
import joblib
filename = 'finalized_model.sav'
model = joblib.load(filename)

#Import python scripts
from preprocessing import preprocess

def main():
    #Setting Application title
    st.title('Fenyx Titanic Model App')

      #Setting Application description
    st.markdown("""
     :dart:  This Streamlit app is made to predict Titanic use case.
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('App.png')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Titanic use case')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        st.info("Input data below")
        #Based on our optimal features selection
        st.subheader("Passenger data")
        age = st.number_input('Age of Passenger ', min_value=0, max_value=75, value=20)
        sibSp = st.number_input('number of siblings/spouses', min_value=0, max_value=8, value=0)
        parch = st.number_input('number of parents/children' , min_value=0, max_value=6, value=0)
        fare = st.number_input('Ticket Price', min_value=0, max_value=512, value=15)
        
        st.subheader("Passenger Title")                       
        title_0 = st.selectbox('Is Passenger title Master:', ('No','Yes'))
        title_1 = st.selectbox('Is Passenger title Mrs:', ('No','Yes'))
        title_2 = st.selectbox('Is Passenger title Mr:', ('No','Yes'))
        title_3 = st.selectbox('Is Passenger title Others:', ('No','Yes'))
        
        st.subheader("Passenger Family Data")   
        fsize = st.number_input('Passenger Family Size', min_value=0, max_value=9, value=1)
        family_size_0 = st.selectbox('Is Passenger Family Size smaller then 5', ('No','Yes'))
        family_size_1 = st.selectbox('Is Passenger Family Size greater then 5', ('No','Yes'))                       
        
        st.subheader("Passenger Embark")   
        embarked_c = st.selectbox('Is Passenger Embarked Cherbourg', ('No','Yes')) 
        embarked_q = st.selectbox('Is Passenger Embarked Queenstown', ('No','Yes')) 
        embarked_s = st.selectbox('Is Passenger Embarked Southampton', ('No','Yes'))
        
        st.subheader("Passenger Ticket")   
        t_A5 = st.selectbox('Is Passenger Ticket starts A5', ('No','Yes')) 
        t_C = st.selectbox('Is Passenger Ticket starts C', ('No','Yes')) 
        t_SOPP = st.selectbox('Is Passenger Ticket starts SOPP', ('No','Yes')) 
        t_STONO = st.selectbox('Is Passenger Ticket starts STONO', ('No','Yes'))                     
        t_SWPP = st.selectbox('Is Passenger Ticket starts SWPP', ('No','Yes'))
        
        st.subheader("Passenger Class")   
        pclass_1 = st.selectbox('Is Passenger Class 1', ('No','Yes')) 
        pclass_2 = st.selectbox('Is Passenger Class 2', ('No','Yes')) 
        pclass_3 = st.selectbox('Is Passenger Class 3', ('No','Yes')) 
        
        sex_female = st.selectbox('Is Passenger Female', ('No','Yes')) 
        sex_male  = st.selectbox('Is Passenger Male', ('No','Yes')) 
        


        
        data = {
                'Age': age,
                'Sibsp': sibSp,
                'Parch':parch,
                'Fare': fare,
                'Title_0': title_0,
                'Title_1': title_1,
                'Title_2': title_2,
                'Title_3': title_3,
                'Fsize': fsize,
                'family_size_0': family_size_0,
                'family_size_1': family_size_1,
                'Embarked_C': embarked_c,
                'Embarked_Q': embarked_q,
                'Embarked_S': embarked_s,
                'T_A': 'No',
                'T_A4':'No',
                'T_A5':t_A5,
                'T_AQ3':'No', 
                'T_AQ4':'No', 
                'T_AS':'No', 
                'T_C':t_C, 
                'T_CA':'No', 
                'T_CASOTON':'No', 
                'T_FC':'No', 
                'T_FCC':'No',
                'T_Fa':'No',
                'T_LINE':'No', 
                'T_LP':'No', 
                'T_PC':'No',
                'T_PP':'No', 
                'T_PPP':'No', 
                'T_SC':'No', 
                'T_SCA3':'No',
                'T_SCA4':'No', 
                'T_SCAH':'No', 
                'T_SCOW':'No', 
                'T_SCPARIS':'No', 
                'T_SCParis':'No', 
                'T_SOC':'No',
                'T_SOP':'No',
                'T_SOPP':t_SOPP, 
                'T_SOTONO2':'No', 
                'T_SOTONOQ':'No', 
                'T_SP':'No', 
                'T_STONO':t_STONO,
                'T_STONO2':'No',
                'T_STONOQ':'No', 
                'T_SWPP':t_SWPP, 
                'T_WC':'No', 
                'T_WEP':'No', 
                'T_x':'No', 
                'Pclass_1':pclass_1,
                'Pclass_2':pclass_2, 
                'Pclass_3':pclass_3, 
                'Sex_female':sex_female, 
                'Sex_male':sex_male
                
            
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)


        #Preprocess inputs
        preprocess_df = preprocess(features_df, 'Online')

        prediction = model.predict(preprocess_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the passenger survive.')
            else:
                st.success('No, the passenger died')
        

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file,encoding= 'utf-8')
            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess_df = preprocess(data, "Batch")
            if st.button('Predict'):
                #Get batch prediction
                prediction = model.predict(preprocess_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the passenger survive.', 0:'No, the passenger died'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
            
if __name__ == '__main__':
        main()
