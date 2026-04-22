import json
import re
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_DIR = BASE_DIR / "data"

TOPIC_MODEL_PATH = MODEL_DIR / "topic_model_lda.pkl"
TOPIC_VECTORIZER_PATH = MODEL_DIR / "topic_vectorizer.pkl"
TOPIC_LABELS_PATH = MODEL_DIR / "topic_labels.json"


def ensure_nltk_data() -> None:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("vader_lexicon", quiet=True)


def load_text_data() -> pd.Series:
    candidates = [
        DATA_DIR / "processed_scaled_down_reviews_with_topics.csv",
        DATA_DIR / "processed_scaled_down_reviews.csv",
        DATA_DIR / "202511-ft_bi1_bi2_course_evaluation.csv",
    ]

    for file_path in candidates:
        if not file_path.exists():
            continue

        data = pd.read_csv(file_path)

        if "text" in data.columns:
            return data["text"].dropna().astype(str)

        # Fallback for course evaluation dataset that uses long free-text column names.
        text_like_cols = [
            col
            for col in data.columns
            if col.lower().startswith("f_") and ("write" in col.lower() or "opinion" in col.lower())
        ]
        if text_like_cols:
            joined = data[text_like_cols].fillna("").astype(str).agg(" ".join, axis=1)
            return joined

    raise FileNotFoundError(
        "No dataset with a usable text column was found in ./data. "
        "Expected one of: processed_scaled_down_reviews_with_topics.csv, "
        "processed_scaled_down_reviews.csv, 202511-ft_bi1_bi2_course_evaluation.csv"
    )


@st.cache_resource(show_spinner=False)
def load_resources():
    ensure_nltk_data()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    sentiment_analyzer = SentimentIntensityAnalyzer()

    topic_labels = {}

    if TOPIC_MODEL_PATH.exists() and TOPIC_VECTORIZER_PATH.exists():
        lda_model = joblib.load(TOPIC_MODEL_PATH)
        vectorizer = joblib.load(TOPIC_VECTORIZER_PATH)

        if TOPIC_LABELS_PATH.exists():
            with TOPIC_LABELS_PATH.open("r", encoding="utf-8") as file:
                topic_labels = {int(k): v for k, v in json.load(file).items()}

        if not topic_labels:
            topic_labels = build_topic_labels(lda_model, vectorizer)

        model_source = "Loaded pretrained topic artifacts from ./model"
        return stop_words, lemmatizer, sentiment_analyzer, lda_model, vectorizer, topic_labels, model_source

    # Lightweight fallback if model artifacts are missing.
    texts = load_text_data().head(5000)
    cleaned_texts = [clean_text(t, stop_words, lemmatizer) for t in texts if str(t).strip()]

    vectorizer = CountVectorizer(max_features=1500)
    x_matrix = vectorizer.fit_transform(cleaned_texts)

    lda_model = LatentDirichletAllocation(
        n_components=5,
        learning_method="batch",
        random_state=42,
        max_iter=12,
    )
    lda_model.fit(x_matrix)

    topic_labels = build_topic_labels(lda_model, vectorizer)
    model_source = "Recreated lightweight LDA model from dataset (fallback mode)"

    return stop_words, lemmatizer, sentiment_analyzer, lda_model, vectorizer, topic_labels, model_source


def clean_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text).lower())
    tokens = word_tokenize(text)
    cleaned_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(cleaned_tokens)


def build_topic_labels(lda_model, vectorizer, top_n: int = 3) -> dict:
    feature_names = vectorizer.get_feature_names_out()
    labels = {}

    for topic_id, topic_weights in enumerate(lda_model.components_):
        top_indices = topic_weights.argsort()[-top_n:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        labels[topic_id] = f"Topic {topic_id + 1}: {', '.join(top_words)}"

    return labels


def predict_topic_and_sentiment(
    text: str,
    stop_words: set,
    lemmatizer: WordNetLemmatizer,
    sentiment_analyzer: SentimentIntensityAnalyzer,
    lda_model,
    vectorizer,
    topic_labels: dict,
):
    cleaned = clean_text(text, stop_words, lemmatizer)
    topic_input = vectorizer.transform([cleaned])
    topic_probs = lda_model.transform(topic_input)[0]

    topic_id = int(np.argmax(topic_probs))
    topic_label = topic_labels.get(topic_id, f"Topic {topic_id + 1}")
    topic_probability = float(topic_probs[topic_id])

    compound_score = sentiment_analyzer.polarity_scores(str(text))["compound"]
    if compound_score >= 0.05:
        sentiment = "positive"
    elif compound_score <= -0.05:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "cleaned_text": cleaned,
        "topic_id": topic_id,
        "topic_label": topic_label,
        "topic_probability": topic_probability,
        "sentiment": sentiment,
        "sentiment_compound": float(compound_score),
    }


def main() -> None:
    st.set_page_config(page_title="Course Evaluation NLP Demo", page_icon="A", layout="centered")

    st.title("Course Evaluation NLP Demo")
    st.write(
        "Enter one student evaluation comment, then click Predict to get the topic (LDA) and sentiment label."
    )

    (
        stop_words,
        lemmatizer,
        sentiment_analyzer,
        lda_model,
        vectorizer,
        topic_labels,
        model_source,
    ) = load_resources()

    st.caption(model_source)

    user_text = st.text_area(
        "Student evaluation text",
        placeholder="Example: The lecturer explains concepts clearly but some labs were too short.",
        height=160,
    )

    if st.button("Predict", type="primary"):
        if not user_text or not user_text.strip():
            st.warning("Please enter evaluation text before prediction.")
            return

        result = predict_topic_and_sentiment(
            text=user_text,
            stop_words=stop_words,
            lemmatizer=lemmatizer,
            sentiment_analyzer=sentiment_analyzer,
            lda_model=lda_model,
            vectorizer=vectorizer,
            topic_labels=topic_labels,
        )

        st.subheader("Prediction Results")
        st.markdown(f"**Predicted Topic:** `{result['topic_label']}`")
        st.markdown(f"**Topic Confidence:** `{result['topic_probability']:.3f}`")
        st.markdown(f"**Predicted Sentiment:** `{result['sentiment'].upper()}`")
        st.markdown(f"**Sentiment Compound Score:** `{result['sentiment_compound']:.3f}`")

        with st.expander("Show preprocessed text"):
            st.code(result["cleaned_text"] or "(empty after preprocessing)")


if __name__ == "__main__":
    main()

