import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def load_dataset():
    """
    Loads the Financial PhraseBank dataset automatically using KaggleHub.
    Falls back to local CSV if offline or fetch fails.
    """
    import os
    import pandas as pd
    import kagglehub

    try:
        print("📦 Downloading Financial PhraseBank dataset via KaggleHub...")
        path = kagglehub.dataset_download("ankurzing/sentiment-analysis-for-financial-news")
        dataset_path = os.path.join(path, "all-data.csv")

        # Read using correct encoding
        df = pd.read_csv(dataset_path, names=["sentiment", "headline"], encoding="ISO-8859-1")
        df["sentiment"] = df["sentiment"].str.strip().str.lower()

        print(f"✅ Loaded {len(df)} samples from Financial PhraseBank (Kaggle).")
        return df

    except Exception as e:
        print(f"⚠️ Could not fetch dataset via KaggleHub: {e}")
        print("📄 Falling back to local dataset...")
        file_path = os.path.join(os.path.dirname(__file__), "../data/processed/all-data.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"❌ Dataset not found at {file_path}. Please create it first.")
        df = pd.read_csv(file_path, names=["sentiment", "headline"])
        df["sentiment"] = df["sentiment"].str.strip().str.lower()
        print(f"✅ Loaded {len(df)} samples from local dataset.")
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
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "../models")
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(MODEL_DIR, "sentiment_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "vectorizer.pkl")

    print(f"DEBUG: Absolute model save path: {os.path.abspath(model_path)}")
    print(f"DEBUG: Current working directory: {os.getcwd()}")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"💾 Model saved at: {model_path}")
    print(f"💾 Vectorizer saved at: {vectorizer_path}")
    print("💾 Model and vectorizer saved in ../models/")

if __name__ == "__main__":
    df = load_dataset()
    train_model(df)
