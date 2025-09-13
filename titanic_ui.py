# -*- coding: utf-8 -*-
"""
Created on Fri Sep 12 13:05:32 2025

@author: 91708
"""
import pickle
import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Load model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Predict function
def predict(PassengerId, Pclass, Age, SibSp, Parch, Fare, Sex, Embarked):
    Sex_male = 1 if Sex == "Male" else 0
    Embarked_Q = 1 if Embarked == "Q" else 0
    Embarked_S = 1 if Embarked == "S" else 0
    features = np.array([[PassengerId, Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S]])
    prediction = model.predict(features)
    proba = model.predict_proba(features)
    return prediction[0], proba[0][1]

# Custom CSS for website feel
st.markdown("""
    <style>
    body {
        background-color: #f4f6f9;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        padding: 1.5em;
        background: linear-gradient(90deg, #007bff, #00c6ff);
        color: white;
        border-radius: 12px;
        margin-bottom: 1.5em;
    }
    .stButton button {
        background: linear-gradient(90deg,#007bff,#00c6ff);
        color: white;
        border-radius: 8px;
        font-size: 18px;
        padding: 0.5em 2em;
    }
    .result-card {
        padding: 20px;
        border-radius: 12px;
        font-size: 20px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_container_width=True)
    st.sidebar.title("⚙️ Settings")
    st.sidebar.info("Enter passenger details to predict survival.")

    # Hero Section
    st.markdown("<div class='title'><h1>🚢 Titanic Survival Predictor</h1><p>Machine Learning powered prediction app</p></div>", unsafe_allow_html=True)

    # Two-column layout
    col1, col2 = st.columns(2)

    with col1:
        PassengerId = st.number_input("Passenger ID", min_value=1, max_value=1000, value=1)
        Pclass = st.selectbox("Passenger Class", [1, 2, 3])
        Age = st.number_input("Age", min_value=0, max_value=100, value=25)
        SibSp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)

    with col2:
        Parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
        Fare = st.number_input("Fare (Ticket Price)", min_value=0.0, max_value=600.0, value=30.0)
        Sex = st.radio("Sex", ["Male", "Female"])
        Embarked = st.selectbox("Embarked Port", ["C", "Q", "S"])

    # Predict button
    if st.button("🔍 Predict Survival"):
        result, prob = predict(PassengerId, Pclass, Age, SibSp, Parch, Fare, Sex, Embarked)

        if result == 1:
            st.markdown(f"<div class='result-card' style='background:#d4edda; color:#155724;'>✅ Passenger SURVIVED <br> 🎯 Probability: {prob:.2%}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-card' style='background:#f8d7da; color:#721c24;'>❌ Passenger did NOT survive <br> 🔎 Probability: {prob:.2%}</div>", unsafe_allow_html=True)

        # Probability Gauge Chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob*100,
            title={'text': "Survival Probability (%)"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green" if result==1 else "red"}}
        ))
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
