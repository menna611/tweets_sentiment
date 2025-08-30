import streamlit as st
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# ---------------------
# Load model & vectorizer
# ---------------------
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ---------------------
# Cleaning function
# ---------------------
def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------
# Streamlit UI
# ---------------------
st.set_page_config(page_title="Twitter Sentiment Classifier", page_icon="🐦", layout="wide")

st.title("🐦 Twitter Sentiment Analysis")
st.markdown("### Enter a tweet below and see its sentiment prediction with confidence scores.")

tweet_input = st.text_area("✍️ Write a tweet:", placeholder="Type something like: I love this new phone!")

if st.button("🔮 Predict"):
    if tweet_input.strip() != "":
        # Clean + Vectorize
        clean_text = clean_tweet(tweet_input)
        vectorized = tfidf.transform([clean_text])
        
        # Predict
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized)[0]

        # Display Result with emoji
        emoji_map = {"Positive":"😍", "Negative":"😡", "Neutral":"😐", "Irrelevant":"🤔"}
        st.markdown(f"### 🎯 Predicted Sentiment: **{prediction} {emoji_map.get(prediction,'')}**")

        # Show probabilities as bar chart
        st.subheader("📊 Prediction Confidence")
        fig, ax = plt.subplots()
        sns.barplot(x=model.classes_, y=proba, palette="viridis", ax=ax)
        ax.set_ylabel("Probability")
        st.pyplot(fig)
    else:
        st.warning("⚠️ Please enter some text!")

# ---------------------
# Sidebar EDA (Optional)
# ---------------------
st.sidebar.title("📊 About This Model")
st.sidebar.info("""
This app uses **TF-IDF + Logistic Regression**  
to classify tweets into **Positive, Negative, Neutral, or Irrelevant**.  

You can extend it by training with **SVM, XGBoost, or BERT** for higher accuracy. 🚀
""")
