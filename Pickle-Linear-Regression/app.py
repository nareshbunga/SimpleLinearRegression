"""Streamlit web app for salary prediction using a trained Linear Regression model."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "linear_regression_model.pkl"
DATA_PATH = ROOT / "data" / "Salary_Data.csv"

st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")
st.title("Salary Prediction")
st.write(
    "Estimate salary from years of experience using a Simple Linear Regression model."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python src/train_model.py` first."
        )
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


model = load_model()

# Use the training data's range to explain when a prediction is extrapolated.
minimum_experience, maximum_experience = 0.0, 50.0
if DATA_PATH.exists():
    training_data = pd.read_csv(DATA_PATH)
    if "YearsExperience" in training_data.columns:
        minimum_experience = float(training_data["YearsExperience"].min())
        maximum_experience = float(training_data["YearsExperience"].max())

st.subheader("Enter your experience")
years_experience = st.number_input(
    "Years of experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5,
)

if st.button("Predict salary", type="primary", use_container_width=True):
    input_data = pd.DataFrame({"YearsExperience": [years_experience]})
    prediction = float(model.predict(input_data)[0])
    st.success(f"Estimated salary: ${prediction:,.2f}")

    if years_experience < minimum_experience or years_experience > maximum_experience:
        st.warning(
            f"This input is outside the training-data range "
            f"({minimum_experience:g}–{maximum_experience:g} years). "
            "The result is an extrapolation and may be less reliable."
        )

st.divider()
st.caption(
    "Educational demonstration. Predictions are estimates from the supplied "
    "dataset and should not be treated as an actual salary offer."
)
