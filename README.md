# AI Stock Sentiment Analyzer

## Overview
The **AI Stock Sentiment Analyzer** is a Python-based application that automatically collects recent financial news headlines, analyzes their sentiment, and produces an overall market outlook for selected stocks.  
This project demonstrates the integration of **Natural Language Processing (NLP)**, **web scraping**, and **machine learning** techniques to extract insight from textual data and build data-driven indicators of market mood.

The system was developed as a self-directed portfolio project to strengthen applied skills in:
- Data collection and preprocessing (BeautifulSoup, pandas)
- Text feature extraction (TF-IDF vectorization)
- Model training and evaluation (Logistic Regression)
- Automation and modular software design in Python

---

## Features
- **Automated Data Collection:** Scrapes live financial headlines from Yahoo Finance with timestamp and source metadata.  
- **End-to-End Sentiment Pipeline:** Trains and persists a TF-IDF + logistic regression classifier for positive / neutral / negative labels.  
- **Batch & Interactive Predictions:** Score new headlines from the command line or trigger analysis inside a Flask web dashboard.  
- **Structured Data Storage:** Stores raw scrapes and model outputs as CSV files in `data/`.  
- **Modular Codebase:** Independent scripts for scraping, training, prediction, and the web UI make it easy to extend or automate.  
- **Portfolio-Ready Documentation:** Includes clear setup steps, usage examples, and future enhancements for continued growth.

---

## Project Architecture
```
AI-Stock-Sentiment-Analyzer/
│
├── app/
│   ├── scraper.py           # Scrape Yahoo Finance headlines and save to data/raw
│   ├── train_model.py       # Download dataset (via KaggleHub) and train the model
│   ├── predict.py           # Run batch predictions on the latest scraped headlines
│   └── web_app.py           # Flask UI to trigger scraping/prediction and view results
│
├── data/
│   ├── raw/                 # Stores unprocessed scraped data
│   └── processed/           # (optional) Cleaned and labeled datasets
│
├── models/                  # Serialized model (.pkl) and vectorizer
├── static/                  # Frontend assets for the dashboard
├── templates/               # HTML templates for Flask web interface
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

---

## Technologies Used
| Category | Tools / Libraries |
|-----------|-------------------|
| **Language** | Python 3.13 |
| **Libraries** | requests, BeautifulSoup4, pandas, scikit-learn, joblib |
| **Data Storage** | CSV (with planned SQLite integration) |
| **Development Tools** | Git, GitHub, PowerShell, Visual Studio Code |
| **OS Environment** | Windows 10 / 11 |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/WaleedOfCarleton/AI-Stock-Sentiment-Analyzer.git
cd AI-Stock-Sentiment-Analyzer
```

### 2. (Optional) Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have `requirements.txt` yet, you can create it by running:
```bash
pip freeze > requirements.txt
```

---

## Usage

### One-time: Train (or retrain) the sentiment model
```bash
python app/train_model.py
```

This downloads the Financial PhraseBank dataset via KaggleHub (if online), cleans the labels, splits the data, trains the logistic regression classifier, and saves both the model and TF-IDF vectorizer to `models/`.

If the Kaggle download fails, place a copy of `all-data.csv` in `data/processed/` and rerun the script.

---

### Collect the latest financial headlines
```bash
python app/scraper.py
```

The script saves a timestamped CSV like `data/raw/news_20251024_1832.csv` containing columns `headline`, `timestamp`, and `source`.

---

### Generate predictions from the command line
```bash
python app/predict.py
```

The script loads the latest raw CSV, applies the trained model, prints the first few predictions, and writes the full results to `data/processed/predictions_<timestamp>.csv`.

---

### Use the interactive Flask dashboard
```bash
python -m app.web_app
```

Open `http://127.0.0.1:5000/` in your browser to:
- Scrape fresh headlines with **Scrape Latest News**
- Run inference with **Analyze Sentiment**
- Review the most recent predictions and sentiment distribution

The app expects `models/sentiment_model.pkl` and `models/vectorizer.pkl` to exist (run the training script first).

---

## Example Output

**Sample CSV Preview (truncated):**
| Headline | Timestamp | Source |
|-----------|------------|---------|
| Dow futures rise after strong earnings reports | 2025-10-24 18:32 | Yahoo Finance |
| Tesla shares drop amid production concerns | 2025-10-24 18:32 | Yahoo Finance |
| S&P 500 gains as inflation fears ease | 2025-10-24 18:32 | Yahoo Finance |

**Model Metrics Example:**
```
precision    recall  f1-score   support

negative       0.88      0.87      0.87      950
neutral        0.89      0.90      0.90     1150
positive       0.91      0.89      0.90     1100

accuracy                           0.89     3200
```

---

## Future Enhancements
- **Real-Time Sentiment Updates:** Automatically re-scrape and re-analyze headlines every hour.  
- **Web Dashboard:** Build an interactive Flask or FastAPI interface for displaying results.  
- **Stock Symbol Filtering:** Map headlines to ticker symbols using NLP keyword matching.  
- **Database Integration:** Use SQLite or PostgreSQL for scalable data management.  
- **Visualization:** Add charts (Plotly / Chart.js) for sentiment over time.  
- **Deployment:** Host the Flask dashboard online using Render or Railway.  

---

## Educational Value
This project serves as a demonstration of practical **machine learning engineering**, highlighting the complete workflow:
1. **Data Acquisition:** Real-world data collection through web scraping.  
2. **Data Preprocessing:** Cleaning, tokenization, and feature extraction.  
3. **Model Building:** Logistic regression for text classification.  
4. **Model Evaluation:** Accuracy and F1 metrics using scikit-learn.  
5. **Software Design:** Modular code architecture, Git-based version control, and documentation.  

---

## Troubleshooting & Tips
- If requests to Yahoo Finance fail, try changing your **User-Agent** header in `scraper.py`.  
- Ensure that all directories exist (`data/raw`, `models/`) before running scripts.  
- Use `python -m pip install --upgrade pip` if dependency installation fails.  
- You can re-train your model at any time by deleting the old `.pkl` files and re-running `train_model.py`.  

---

## Author
**Waleed Abu-Osbeh**  
Bachelor of Computer Science (Honours, Co-op) — Carleton University  
Email: [waleedabuosbeh@cmail.carleton.ca](mailto:waleedabuosbeh@cmail.carleton.ca)  
LinkedIn: [linkedin.com/in/waleed-abu-osbeh-525a81253](https://www.linkedin.com/in/waleed-abu-osbeh-525a81253)

---

## License
This project is released under the [MIT License](https://opensource.org/licenses/MIT).

---

## Acknowledgments
- Financial PhraseBank dataset (Kaggle, Ankur Zing)  
- Yahoo Finance for real-time market news  
- Scikit-learn documentation and examples  

---

## Version History
| Version | Date | Description |
|----------|------|-------------|
| 1.0 | Oct 2025 | Initial release with scraping and training functionality |
| 1.1 | (planned) | Add prediction and Flask dashboard integration |
| 1.2 | (planned) | Implement continuous retraining and live visualization |
