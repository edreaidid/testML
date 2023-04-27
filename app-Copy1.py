import streamlit as st
import pandas as pd
import joblib

# Title
st.header("Glucose Control Prediction App")

# Input bar 1
sbp = st.number_input("Enter sbp")

# Input bar 2
dbp = st.number_input("Enter dbp")

# Dropdown input
smoking = st.selectbox("smoking", ("Yes", "No"))

# If button is pressed
if st.button("Submit"):
    
    # Unpickle classifier
    clf = joblib.load("C:\\Users\\raef\\Desktop\\ML python\\simple ML stapp\\hstat.pkl")
    
    # Store inputs into dataframe
    X = pd.DataFrame([[sbp, dbp, smoking]], 
                     columns = ["sbp", "dbp", "smoking"])
    X = X.replace(["Yes", "No"], [1, 0])
    
    # Get prediction
    prediction = clf.predict(X)[0]
    
    # Output prediction
    st.text(f"Your glucose control will be {prediction}")

