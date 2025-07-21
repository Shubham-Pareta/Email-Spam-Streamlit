import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

model = joblib.load("models/spam_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

st.set_page_config(page_title="📧 Email Spam Detector", page_icon="📨", layout="centered")

st.markdown(
    """
    <style>
        .main-title {
            text-align: center;
            font-size: 2.5em;
            color: #0F62FE;
            margin-bottom: 0.5em;
        }
        .subtext {
            text-align: center;
            font-size: 1.1em;
            color: #444;
            margin-bottom: 2em;
        }
        .result-card {
            padding: 1.5em;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: bold;
            text-align: center;
            box-shadow: 0 0 10px rgba(0,0,0,0.08);
            margin-top: 2em;
        }
    </style>
    <h1 class='main-title'>📧 Email Spam Classifier</h1>
    <p class='subtext'>Paste an email below and click <strong>Classify</strong> to detect spam.</p>
    """,
    unsafe_allow_html=True
)

with st.container():
    input_text = st.text_area(
        "✍️ Paste Email Content Here:",
        height=200,
        placeholder="e.g. Congratulations! You've won a free iPhone! Click here to claim it..."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    center_btn = st.columns(3)
    with center_btn[1]:
        classify = st.button("🚀 Classify Email")

    if classify:
        if input_text.strip() == "":
            st.warning("⚠️ Please enter some email text first.")
        else:
            cleaned = clean_text(input_text)
            vectorized = vectorizer.transform([cleaned])
            prediction = model.predict(vectorized)
            proba = model.predict_proba(vectorized)[0][prediction[0]]

            st.markdown("---")
            if prediction[0] == 1:
                st.markdown(
                    f"<div class='result-card' style='background-color:#FFE6E6; color:#D70040;'>🚫 SPAM detected with {proba:.2%} confidence.</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result-card' style='background-color:#E6FFE6; color:#007E33;'>✅ Not Spam with {proba:.2%} confidence.</div>",
                    unsafe_allow_html=True
                )

            st.progress(proba)

st.markdown("<br><hr>", unsafe_allow_html=True)
