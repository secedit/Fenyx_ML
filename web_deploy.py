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
        t_A = st.selectbox('Is Passenger Ticket starts A', ('No','Yes')) 
        t_A4 = st.selectbox('Is Passenger Ticket starts A4', ('No','Yes'))
        t_A5 = st.selectbox('Is Passenger Ticket starts A5', ('No','Yes')) 
        t_AQ3 = st.selectbox('Is Passenger Ticket starts AQ3', ('No','Yes'))
        t_AQ4 = st.selectbox('Is Passenger Ticket starts AQ4', ('No','Yes')) 
        t_AS = st.selectbox('Is Passenger Ticket starts AS', ('No','Yes')) 
        t_C = st.selectbox('Is Passenger Ticket starts C', ('No','Yes')) 
        t_CA = st.selectbox('Is Passenger Ticket starts CA', ('No','Yes')) 
        t_CASOTON = st.selectbox('Is Passenger Ticket starts CASOTON', ('No','Yes')) 
        t_FC = st.selectbox('Is Passenger Ticket starts FC', ('No','Yes')) 
        t_FCC = st.selectbox('Is Passenger Ticket starts FCC', ('No','Yes')) 
        t_Fa =  st.selectbox('Is Passenger Ticket starts Fa', ('No','Yes')) 
        t_LINE =  st.selectbox('Is Passenger Ticket starts LINE', ('No','Yes')) 
        t_LP=  st.selectbox('Is Passenger Ticket starts LP', ('No','Yes')) 
        t_PC =  st.selectbox('Is Passenger Ticket starts PC', ('No','Yes'))
        t_PP =  st.selectbox('Is Passenger Ticket starts PP', ('No','Yes'))
        t_PPP =  st.selectbox('Is Passenger Ticket starts PPP', ('No','Yes')) 
        t_SC =  st.selectbox('Is Passenger Ticket starts SC', ('No','Yes')) 
        t_SCA3 = st.selectbox('Is Passenger Ticket starts SCA3', ('No','Yes')) 
        t_SCA4 = st.selectbox('Is Passenger Ticket starts SCA4', ('No','Yes')) 
        t_SCAH = st.selectbox('Is Passenger Ticket starts SCAH', ('No','Yes')) 
        t_SCOW = st.selectbox('Is Passenger Ticket starts SCOW', ('No','Yes'))
        t_SCPARIS = st.selectbox('Is Passenger Ticket starts SCPARIS', ('No','Yes')) 
        t_SCParis = st.selectbox('Is Passenger Ticket starts SCParis', ('No','Yes')) 
        t_SOC = st.selectbox('Is Passenger Ticket starts SOC', ('No','Yes')) 
        t_SOP = st.selectbox('Is Passenger Ticket starts SOP', ('No','Yes')) 
        t_SOPP = st.selectbox('Is Passenger Ticket starts SOPP', ('No','Yes')) 
        t_SOTONO2 = st.selectbox('Is Passenger Ticket starts SOTONO2', ('No','Yes')) 
        t_SOTONOQ = st.selectbox('Is Passenger Ticket starts SOTONOQ', ('No','Yes'))
        t_SP = st.selectbox('Is Passenger Ticket starts SP', ('No','Yes')) 
        t_STONO = st.selectbox('Is Passenger Ticket starts STONO', ('No','Yes')) 
        t_STONO2 = st.selectbox('Is Passenger Ticket starts STONO2', ('No','Yes'))
        t_STONOQ = st.selectbox('Is Passenger Ticket starts STONOQ', ('No','Yes'))                       
        t_SWPP = st.selectbox('Is Passenger Ticket starts SWPP', ('No','Yes'))
        t_WC= st.selectbox('Is Passenger Ticket starts WC', ('No','Yes'))
        t_WEP = st.selectbox('Is Passenger Ticket starts WEP', ('No','Yes'))
        t_x = st.selectbox('Is Passenger Ticket starts others', ('No','Yes'))                     
                             
        
                             
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
                'T_A': t_A,
                'T_A4':t_A4,
                'T_A5':t_A5,
                'T_AQ3':t_AQ3, 
                'T_AQ4':t_AQ4, 
                'T_AS':t_AS, 
                'T_C':t_C, 
                'T_CA':t_CA, 
                'T_CASOTON':t_CASOTON, 
                'T_FC':t_FC, 
                'T_FCC':t_FCC,
                'T_Fa':t_Fa,
                'T_LINE':t_LINE, 
                'T_LP':t_LP, 
                'T_PC':t_PC,
                'T_PP':t_PP, 
                'T_PPP':t_PPP, 
                'T_SC':t_SC, 
                'T_SCA3':t_SCA3,
                'T_SCA4':t_SCA4, 
                'T_SCAH':t_SCAH, 
                'T_SCOW':t_SCOW, 
                'T_SCPARIS':t_SCPARIS, 
                'T_SCParis':t_SCParis, 
                'T_SOC':t_SOC,
                'T_SOP':t_SOP,
                'T_SOPP':t_SOPP, 
                'T_SOTONO2':t_SOTONO2, 
                'T_SOTONOQ':t_SOTONOQ, 
                'T_SP':t_SP, 
                'T_STONO':t_STONO,
                'T_STONO2':t_STONO2,
                'T_STONOQ':t_STONOQ, 
                'T_SWPP':t_SWPP, 
                'T_WC':t_WC, 
                'T_WEP':t_WEP, 
                'T_x':t_x, 
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
                prediction_df = prediction_df.replace({1:'Yes, the passenger survive.', 
                                                    0:'No, the passenger died'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)
            
if __name__ == '__main__':
        main()
