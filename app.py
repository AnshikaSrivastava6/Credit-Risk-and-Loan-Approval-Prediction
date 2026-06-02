import streamlit as st
import numpy as np
import joblib
model=joblib.load( "loan_approval_model.pkl")
scaler=joblib.load( "scaler.pkl")
st.title("CREDIT RISK AND LOAN APPROVAL PREDICTION")
age=st.number_input("Enter Age",min_value=18,max_value=100)
income=st.number_input("Enter income")
loan_amount=st.number_input("Enter loan amount")
years_of_experience=st.number_input("Enter year of experience")
credit_score=st.number_input("Enter credit score")

gender=st.selectbox("Gender",["Male","Female"])
education = st.selectbox("Education",["High School","Bachelors","Masters","PhD"])
city = st.selectbox( "City", ["Houston", "New York", "San Francisco"])
employment = st.selectbox("Employment Type",["Salaried", "Self-Employed", "Unemployed"])

if st.button("predict"):
    gender_val = 1 if gender == "Female" else 0
    education_map = {
        "High School": 1,
        "Bachelors": 2,
        "Masters": 3,
        "PhD": 4}
       
    education_val = education_map[education]

    
    city_houston = 1 if city == "Houston" else 0
    city_newyork = 1 if city == "New York" else 0
    city_sf = 1 if city == "San Francisco" else 0

    emp_self = 1 if employment == "Self-Employed" else 0
    emp_unemp = 1 if employment == "Unemployed" else 0
    data = np.array([[
        age,
        income,
        loan_amount,
        credit_score,
        years_of_experience,
        gender_val,
        education_val,
        city_houston,
        city_newyork,
        city_sf,
        emp_self,
        emp_unemp
         ]])
    data = scaler.transform(data)

    
    prediction = model.predict(data)

     # Probabilities
    probs = model.predict_proba(data)

    rejection_prob = probs[0][0] * 100
    approval_prob = probs[0][1] * 100

    if prediction[0] == 1:
            st.success("Loan Approved")
    else:
            st.error("Loan Rejected")
 
    st.write(f"Approval Probability: {approval_prob:.2f}%")
    st.write(f"Rejection Probability: {rejection_prob:.2f}%")
    st.write(f"Estimated Credit Risk: {rejection_prob:.2f}%")                
                                   
