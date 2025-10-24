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
- **Automated Data Collection:** Scrapes live financial headlines from Yahoo Finance.  
- **Structured Data Storage:** Saves scraped data into organized CSV files for reuse and reproducibility.  
- **Machine Learning Classification:** Trains a supervised sentiment classifier (positive / neutral / negative) using a labeled dataset.  
- **Model Persistence:** Saves trained models (`.pkl` files) for future predictions.  
- **Extendable Framework:** Ready for integration with Flask for visualization and API endpoints.  
- **Clear Code Organization:** Separation of concerns across `scraper.py`, `train_model.py`, and future modules for prediction and visualization.

---

## Project Architecture
```
AI-Stock-Sentiment-Analyzer/
│
├── app/
│   ├── scraper.py           # Scrapes Yahoo Finance headlines
│   ├── train_model.py       # Trains logistic regression sentiment model
│   ├── predict.py           # (future) Predicts sentiment of new scraped data
│   └── dashboard.py         # (future) Flask-based web dashboard for visualization
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

### Step 1 — Scrape financial headlines
Run the scraping module to collect the latest market headlines:

```bash
python app/scraper.py
```

**Output Example:**
```
✅ Saved 50 headlines to ../data/raw/news_20251024_1832.csv
```

The resulting CSV file will be located under `data/raw/` and contain columns:
- `headline`
- `timestamp`
- `source`

---

### Step 2 — Train the sentiment model
Train a logistic regression classifier using a labeled financial dataset:

```bash
python app/train_model.py
```

**Example Console Output:**
```
✅ Loaded 4,800 labeled samples.
Model accuracy: 89.7%
💾 Model and vectorizer saved in ../models/
```

After completion, the trained model and TF-IDF vectorizer are stored in:
```
models/sentiment_model.pkl
models/vectorizer.pkl
```

---

### Step 3 — (Planned) Predict sentiment of new data
In the future `predict.py` module, the saved model will be used to:
- Load new headlines from `data/raw/`
- Apply preprocessing and vectorization
- Predict sentiment for each headline
- Output results with sentiment labels and confidence scores

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
