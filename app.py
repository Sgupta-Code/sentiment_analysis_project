import streamlit as st
from sentiment_analyzer import SentimentAnalyzer
import pandas as pd
import pickle
import nltk

nltk.download('punkt_tab')
nltk.download('stopwords')


@st.cache_resource
def load_model():
    analyzer = pickle.load(open('sentiment_model.pkl', 'rb'))
    return analyzer

def main():
    st.title("Sentiment Analysis App")

    st.sidebar.header("About")
    st.sidebar.info("This app analysis the sentiment of text using machine learning.")

    text_input = st.text_area("Enter text to analyze:",height = 150)

    if st.button("Analyze Sentiment"):
        if text_input:
            analyzer = load_model()

            sentiment, confidence = analyzer.predict_sentiment(text_input)

            st.subheader("Analysis Result")

            if sentiment == 1:
                st.success(f"Positive Sentiment (Confidence: {confidence:.2f})")
            else:
                st.success(f"Negative Sentiment (Confidence: {confidence:.2f})")

            st.subheader("Processed Text")
            st.write(analyzer.preprocess_text(text_input))

        else:
            st.warning("PLease enter some text to analyze.")

    st.subheader("Try these examples:")
    examples = [
        "This product is amazing! I love it so much.",
        "The service was terrible and the staff was rude.",
        "The movie was okay, not great but not bad either."
    ]

    for example in examples:
        if st.button(example[:50] + "..."):
            st.text_area("Example text:",value=example,height=100)

if __name__ == "__main__":
    main()
