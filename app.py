import streamlit as st
import joblib
import re
from pathlib import Path

# Find project folder
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
MODEL_PATH = BASE_DIR / "sentiment_model.pkl"

model = joblib.load(MODEL_PATH)


# Text preprocessing
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# Page configuration
st.set_page_config(
    page_title="MARI AI",
    page_icon="🤖",
    layout="centered"
)


# Application title
st.title("🤖 MARI AI")
st.subheader("Customer Feedback Sentiment Analyzer")

st.write(
    "Enter a customer review and let AI analyze its sentiment."
)


# Customer input
review = st.text_area(
    "Enter Customer Review:",
    placeholder="Example: Hotel food is very good..."
)


# Prediction button
if st.button("Analyze Review"):

    if review.strip():

        cleaned_review = clean_text(review)

        prediction = model.predict([cleaned_review])[0]

        st.success("Analysis Completed!")

        if prediction == "Positive":
            st.success("😊 Sentiment: Positive")

        elif prediction == "Negative":
            st.error("😞 Sentiment: Negative")

        else:
            st.info("😐 Sentiment: Neutral")

    else:
        st.warning("Please enter a customer review!")


st.divider()

st.caption("MARI AI | NLP Mini Project")