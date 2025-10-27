import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os

def scrape_yahoo_finance(num_articles=50):
    """
    Scrapes the latest financial news headlines from Yahoo Finance.
    Returns a pandas DataFrame with columns: ['headline', 'timestamp', 'source'].
    """
    url = "https://finance.yahoo.com/topic/stock-market-news/"
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(response.text, "html.parser")

    headlines = []
    articles = soup.find_all("h3", limit=num_articles)

    for article in articles:
        text = article.get_text().strip()
        if text:
            headlines.append({
                "headline": text,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source": "Yahoo Finance"
            })

    df = pd.DataFrame(headlines)
    return df


def save_to_csv(df):
    """Saves scraped data to data/raw/news_<timestamp>.csv"""
    os.makedirs("../data/raw", exist_ok=True)
    file_path = f"../data/raw/news_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} headlines to {file_path}")
    return file_path


def main():
    """Combined function that scrapes and saves data."""
    df = scrape_yahoo_finance()
    file_path = save_to_csv(df)
    return file_path


# Allow running this script directly (for testing)
if __name__ == "__main__":
    main()
