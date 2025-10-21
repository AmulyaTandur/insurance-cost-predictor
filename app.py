
#Now, you will have a file called "app.py" saved in the working directory.

import numpy as np
import pandas as pd
import streamlit as st
import pickle 

model=pickle.load(open('insurance_model.pkl','rb'))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define a function to create Dataframe  with user inputs - 

def preprocess_input(age,sex, bmi, children, smoker):
    data = pd.DataFrame({"age": [age],
                         "sex": [sex],
                         "bmi": [bmi],
                         "children":[children],
                         "smoker": [smoker] })
    return data


# Define a function to predict Charges -
def predict_insurance_charges(data):
    data_scaled = scaler.transform(data)     # Apply scaler if model was trained on scaled inputs
    prediction = model.predict(data_scaled)
    return prediction

# Main app function
def main():
    st.title('Insurance Charges Prediction App')
    st.write("Enter customer details to predict insurance charges")
    st.sidebar.title('User Input')
    
    age = st.sidebar.slider("Age",20,100, step =1,value =30)    # step=1 → The slider increases/decreases in steps of 1 year.
                                                                 # value=30 → The default selected value is 30.
    sex = st.sidebar.selectbox("Sex",[0,1], format_func =lambda x: "Female" if x==0 else "Male")
    bmi = st.sidebar.slider("BMI", 10.0, 40.0, step =0.1, value =20.0)
    children = st.sidebar.slider('Number of Children', 0,10, step =1, value =0)
    smoker = st.sidebar.selectbox("Smoker",[0,1], format_func =lambda x: "No" if x==0 else "Yes")

    if st.button("Predict Charges"):
        input_data = preprocess_input(age,sex, bmi, children, smoker)
        prediction = predict_insurance_charges(input_data)
        
        st.subheader('Estimated Insurance Charge :')
        st.success(f"${float(prediction[0]):,.2f}")        # formats the number(i.e.,charges) with commas and 2 decimal places.

if __name__ == "__main__":
    main()
