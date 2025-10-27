import os
import sys
import glob
from datetime import datetime

import pandas as pd
import joblib
from flask import Flask, redirect, url_for, flash, render_template_string

# Make project root importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Robust imports for scraper/predict utilities
try:
    from app.scraper import main as run_scraper
except ImportError:
    from scraper import main as run_scraper  # when running without package

# Paths
RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")
os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load model/vectorizer once
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

app = Flask(__name__)
app.secret_key = "dev"  # for flash messages

def analyze_latest() -> str | None:
    """Analyze latest raw CSV and save to data/processed/predictions_<timestamp>.csv"""
    csv_files = sorted(glob.glob(os.path.join(RAW_PATH, "news_*.csv")), key=os.path.getmtime)
    if not csv_files:
        return None
    latest_raw = csv_files[-1]
    df = pd.read_csv(latest_raw)
    if "headline" not in df.columns or df.empty:
        return None

    X = vectorizer.transform(df["headline"].astype(str))
    preds = model.predict(X)
    out = pd.DataFrame({"headline": df["headline"], "predicted_sentiment": preds})

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = os.path.join(PROCESSED_PATH, f"predictions_{ts}.csv")
    out.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

@app.route("/")
def index():
    pred_files = sorted(glob.glob(os.path.join(PROCESSED_PATH, "predictions_*.csv")), key=os.path.getmtime)
    latest_label = "None"
    table_html = "<p class='text-muted'>No predictions yet. Click Analyze Sentiment.</p>"
    summary_html = ""

    if pred_files:
        latest_pred = pred_files[-1]
        latest_label = os.path.basename(latest_pred)
        df = pd.read_csv(latest_pred)

        counts = df["predicted_sentiment"].value_counts()
        total = int(counts.sum())
        pos = int(counts.get("positive", 0))
        neu = int(counts.get("neutral", 0))
        neg = int(counts.get("negative", 0))

        summary_html = f"""
        <div class="alert alert-light border">
            <strong>Total:</strong> {total} |
            <span class="text-success">Positive: {pos}</span> |
            <span class="text-secondary">Neutral: {neu}</span> |
            <span class="text-danger">Negative: {neg}</span>
        </div>
        """
        table_html = df.head(50).to_html(classes="table table-striped table-sm", index=False)

    template = """
    <html>
    <head>
        <title>AI Stock Sentiment Analyzer</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
    </head>
    <body class="p-4">
        <div class="container">
            <h1 class="mb-4">📊 AI Stock Sentiment Analyzer</h1>

            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="alert alert-{{category}}">{{ message }}</div>
                {% endfor %}
              {% endif %}
            {% endwith %}

            <div class="mb-3">
                <a href="{{ url_for('scrape') }}" class="btn btn-primary me-2">Scrape Latest News</a>
                <a href="{{ url_for('analyze') }}" class="btn btn-success">Analyze Sentiment</a>
            </div>

            {{ summary_html|safe }}
            <h5 class="mt-4">Recent Headlines (sample of 50)</h5>
            {{ table_html|safe }}
            <p class="text-muted mt-2">Latest predictions file: {{ latest_label }}</p>
        </div>
    </body>
    </html>
    """
    return render_template_string(
        template,
        summary_html=summary_html,
        table_html=table_html,
        latest_label=latest_label,
    )

@app.route("/scrape")
def scrape():
    try:
        run_scraper()
        flash("Scraped latest headlines.", "success")
    except Exception as e:
        flash(f"Scrape failed: {e}", "danger")
    return redirect(url_for("index"))

@app.route("/analyze")
def analyze():
    out_path = analyze_latest()
    if not out_path:
        flash("No raw news to analyze. Click Scrape first.", "warning")
    else:
        flash(f"Saved predictions to {os.path.basename(out_path)}", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    # Run from project root: python -m app.web_app
    # Or from app folder:    python web_app.py
    app.run(debug=True)