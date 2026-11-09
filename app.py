import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "wine_model.pkl"
DATA_PATH = ROOT / "wine_quality.csv"

st.title("🍷 Wine Quality Prediction System")
st.caption(
    "Section B classroom prototype for preliminary red-wine quality classification. "
    "Predictions are screening aids, not final quality-control decisions."
)

if not MODEL_PATH.exists():
    st.error("Required file not found: wine_model.pkl. Run `python train_model.py` in this project folder first.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Required file not found: wine_quality.csv. Run `python train_model.py` in this project folder first.")
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)

model = load_model(str(MODEL_PATH))
data = pd.read_csv(DATA_PATH)

c1, c2, c3 = st.columns(3)
c1.metric("Records", len(data))
c2.metric("Average quality score", round(data["quality"].mean(), 2))
c3.metric("Good-quality records", int((data["quality"] >= 6).sum()))

st.subheader("Dataset preview")
st.dataframe(data.head(10), use_container_width=True)

st.subheader("Predict uploaded wine samples")
st.write("Upload a CSV containing the same 11 predictor columns used to train the model.")

required_columns = list(model.feature_names_in_)
template = data[required_columns].head(5)
st.download_button(
    "Download sample input CSV",
    template.to_csv(index=False),
    "sample_wine_input.csv",
    "text/csv"
)

uploaded = st.file_uploader("Choose wine sample CSV", type=["csv"])

if uploaded is not None:
    try:
        new_data = pd.read_csv(uploaded)
        missing = [column for column in required_columns if column not in new_data.columns]

        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            predictions = model.predict(new_data[required_columns])
            results = new_data.copy()
            results["Prediction"] = [
                "GOOD QUALITY" if value == 1 else "NEEDS IMPROVEMENT"
                for value in predictions
            ]

            st.subheader("Prediction results")
            st.dataframe(results, use_container_width=True)
            st.metric("Predicted good-quality samples", int((predictions == 1).sum()))
            st.download_button(
                "Download prediction results",
                results.to_csv(index=False),
                "wine_quality_predictions.csv",
                "text/csv"
            )
    except Exception as error:
        st.error(f"Could not read or process the uploaded CSV: {error}")

st.info(
    "Responsible use: This model is trained on a public red-wine dataset. "
    "It is not a substitute for laboratory testing, sensory evaluation, food-safety "
    "procedures, or professional quality-control decisions."
)
