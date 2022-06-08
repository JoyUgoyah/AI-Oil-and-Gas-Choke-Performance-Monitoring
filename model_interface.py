import pickle
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
import time



# Insert project title as header
st.title('Artificial Intelligence-based Model for Reliability Centered Maintenance of Oil and Gas Marginal Field Facility')
st.markdown('By Olawale Adenuga [G2019/PHD/EM/METI/CETM/FT/001]')


# Data input description
st.subheader('Input your data in the format described below')
st.markdown('Please upload an excel file of `.csv` format containing the following parameters in the correct oilfield units as indicated to avoid prediction error.')
st.markdown('Start date (yyyy-mm-dd): The date of well test data recording.')
st.markdown('Choke Size (/64"): The size of choke being used for hydrocarbon production.')
st.markdown('FTHP (psig): Flowing Tubing Head Pressure.')
st.markdown('FLP (psig): Flow Line Pressure.')
st.markdown('FLT (°F): Flow Line Temperature.')
st.markdown('Oil T (°F): Oil Temperature.')
st.markdown('Gas T (°F): Gas Temperature.')
st.markdown('Gas P (psig): Gas Pressure.')
st.markdown('Gas Diff (Inwg): Gas Difference.')
st.markdown('Oil Rate (bbls/d): Volume of oil produced per day.',)
st.markdown('Gas Rate (mmscf/d): Volume of gas produced per day.')
st.markdown('Water Rate (bbls/d): Volume of water being produced alongside oil and gas.')
st.markdown('GOR (scf/d): Gas Oil Ratio.')
st.markdown('Oil (bbls): total volume of oil.')
st.markdown('Gas (mmscf): total volume of gas.')
st.markdown('Water (bbls): total volume of water.')
st.markdown('BSW (%): Percentage of Basic Sediments and Water produced.')


# Upload file button widget
st.subheader('Please upload you well test data here')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    well_data = pd.read_csv(uploaded_file)

    def preprocess(well_data=well_data):
        '''
        Preprocess raw input data.

        Args:
        ---
        input_data - Batch of samples. Numpy array of shape (m, 17) where m is
        the number of samples and 17 is the dimension of the feature vector.
        '''
        # drop first row
        well_data.drop(index=0, inplace=True)

        # drop rows with missing values
        well_data.dropna(axis=0, inplace=True)

        # Convert columns with Object data type to float

        well_data['Choke Size'] = well_data['Choke Size'].astype(float, errors = 'raise')
        well_data['FTHP'] = well_data['FTHP'].astype(float, errors = 'raise')
        well_data['FLP'] = well_data['FLP'].astype(float, errors = 'raise')
        well_data['FLT'] = well_data['FLT'].astype(float, errors = 'raise')
        well_data['Oil T'] = well_data['Oil T'].astype(float, errors = 'raise')
        well_data['Gas T'] = well_data['Gas T'].astype(float, errors = 'raise')
        well_data['Gas P'] = well_data['Gas P'].astype(float, errors = 'raise')
        well_data['Gas Diff'] = well_data['Gas Diff'].astype(float, errors = 'raise')
        well_data['Oil Rate'] = well_data['Oil Rate'].astype(float, errors = 'raise')
        well_data['Gas Rate'] = well_data['Gas Rate'].astype(float, errors = 'raise')
        well_data['Water Rate'] = well_data['Water Rate'].astype(float, errors = 'raise')
        well_data['GOR'] = well_data['GOR'].astype(float, errors = 'raise')
        well_data['Oil'] = well_data['Oil'].astype(float, errors = 'raise')
        well_data['Gas'] = well_data['Gas'].astype(float, errors = 'raise')
        well_data['Water'] = well_data['Water'].astype(float, errors = 'raise')
        well_data['BSW'] = well_data['BSW'].astype(float, errors = 'raise')    
    
        return well_data


    # Load a fitted model from the local filesystem into memory.
    filename = open('model.pkl', 'rb')
    model = pickle.load(filename)
     

    def predict_batch(model, batch_input_features):
        '''
        Function that predicts a batch of sampels.

        Args:
        ---
        batch_input_features: A batch of features required by the model to
        generate predictions. Numpy array of shape (m, n) where m is the 
        number of samples and n is the dimension of the feature vector.

        Returns:
        --------
        prediction: Predictions of the model. Numpy array of shape (m,).
        '''
        # Import evaluation metric

        # Make prediction
        
        y_pred = model.predict(batch_input_features)

        # Show model prediction accuracy score and dataframe containing date and prediction
        
 
        return y_pred
    # Preprocess input data   
    preprocess()  

    # Define features
    X = well_data.drop(columns=['Start date'], axis=1) 

    # Make prediction
    y_pred = predict_batch(model=model, batch_input_features=X) 

    # Output data frame
    output = pd.DataFrame({'Start date': well_data['Start date'], 'Prediction': y_pred})

    #Print predictions as table
    st.table(output)
    
    # Give double line spacing
    st.markdown(' ')
    st.markdown(' ')

    # Add line graph of result
    c = alt.Chart(output, title='Monthly Choke Performance Monitoring').mark_line().encode(
     x='Start date', y='Prediction').properties(width=800, height=300)

    st.altair_chart(c, use_container_width=True)