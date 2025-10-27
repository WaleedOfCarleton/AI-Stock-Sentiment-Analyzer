import os
import pandas as pd
import joblib
from datetime import datetime

# --- Paths ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")

# --- Load model and vectorizer ---
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_sentiment(headlines):
    """Predicts sentiment for a list of headlines."""
    X = vectorizer.transform(headlines)
    preds = model.predict(X)
    return list(zip(headlines, preds))

if __name__ == "__main__":
    # --- Find the most recent scraped file ---
    csv_files = [f for f in os.listdir(RAW_DATA_PATH) if f.endswith(".csv")]
    if not csv_files:
        print("❌ No scraped news files found in data/raw/. Run scraper.py first.")
        exit()

    latest_file = max(csv_files, key=lambda f: os.path.getctime(os.path.join(RAW_DATA_PATH, f)))
    latest_path = os.path.join(RAW_DATA_PATH, latest_file)

    print(f"🔍 Analyzing headlines from: {latest_file}")
    df = pd.read_csv(latest_path)

    if "headline" not in df.columns:
        print("❌ No 'headline' column found in the CSV.")
        exit()

    headlines = df["headline"].astype(str).tolist()
    results = predict_sentiment(headlines)

    # --- Display sample output ---
    print("\n🧠 Sentiment Predictions (first 10):")
    for text, label in results[:10]:
        print(f"• {label.upper():<8} — {text}")

    # --- Save full predictions to CSV ---
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(PROCESSED_PATH, f"predictions_{timestamp}.csv")

    df_results = pd.DataFrame(results, columns=["headline", "predicted_sentiment"])
    df_results.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\n💾 Saved all predictions to: {output_path}")
