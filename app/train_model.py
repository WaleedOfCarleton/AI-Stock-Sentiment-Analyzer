import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_dataset():
    """
    Loads a pre-labeled finance sentiment dataset.
    You can replace this with your own labeled CSV later.
    """
    url = "https://raw.githubusercontent.com/ankurzing/sentiment-analysis-for-financial-news/master/all-data.csv"
    df = pd.read_csv(url, names=["sentiment", "headline"])
    df["sentiment"] = df["sentiment"].str.strip()
    print(f"✅ Loaded {len(df)} labeled samples.")
    return df

def train_model(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df["headline"], df["sentiment"], test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print(f"\nModel accuracy: {accuracy_score(y_test, preds)*100:.2f}%")
    print(classification_report(y_test, preds))

    # Save model + vectorizer
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/sentiment_model.pkl")
    joblib.dump(vectorizer, "../models/vectorizer.pkl")
    print("💾 Model and vectorizer saved in ../models/")

if __name__ == "__main__":
    df = load_dataset()
    train_model(df)
