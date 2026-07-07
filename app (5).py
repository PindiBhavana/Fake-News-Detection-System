import streamlit as st
import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.set_page_config(page_title="Fake News Detection", page_icon="📰")

st.title("📰 Fake News Detection")
st.write("Predict whether a news article is **FAKE** or **REAL** using a TF-IDF + Multinomial Naive Bayes model.")

news = st.text_area("Enter a news article", height=220)

if st.button("Predict"):
    if not news.strip():
        st.warning("Please enter a news article.")
    else:
        X = vectorizer.transform([news])
        pred = model.predict(X)[0]
        prob = model.predict_proba(X).max() * 100

        if pred == "REAL":
            st.balloons()
            st.success("Prediction: REAL")
        else:
            st.error("Prediction: FAKE")

        st.info(f"Confidence: {prob:.2f}%")
