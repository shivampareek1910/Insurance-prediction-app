import joblib
import streamlit as st
import numpy as np 

model = joblib.load("insurance_model (1).joblib")
poly = joblib.load("poly_transformer.joblib")

st.header("INSURANCE PREDICTION APP")

age = st.number_input(label = "Enter age:", min_value=10, value=30, max_value=70, step=1)
weight = st.number_input(label= "Enter weight in KG:", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
height_feet = st.number_input(label="Enter height in feet:", min_value=1, max_value=12, value=5)
height_inches = st.number_input(label="Enter height in inches:", min_value=0, max_value=11)
height_mrs = height_feet*0.3048 + height_inches*0.0254
bmi = weight / (height_mrs)**2

children = st.selectbox(
"Select numbers of children",
(0,1,2,3,4,5))

smoker_num = st.selectbox(
"Smoking status",
("yes","no"))

smoker_num = 1 if smoker_num == "Yes" else 0

test_data = [[age, bmi, children, smoker_num]]

if st.button("Submit") == True:
    poly_data = poly.transform(test_data)
    y_pred_sqrt = model.predict(poly_data)[0]
    y_pred = round(y_pred_sqrt**2,2)
    st.markdown(f"### Your insurance premium is ${y_pred}")
