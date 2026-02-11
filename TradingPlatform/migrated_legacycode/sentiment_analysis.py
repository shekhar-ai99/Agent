# File: sentiment_analysis.py
import pandas as pd
import os
from textblob import TextBlob

data_folder = "/mnt/data/"
input_file = os.path.join(data_folder, "nifty_signals.csv")
output_file = os.path.join(data_folder, "nifty_sentiment.csv")

def analyze_sentiment(text):
    """Analyze sentiment of signal"""
    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

if __name__ == "__main__":
    df = pd.read_csv(input_file)
    df['sentiment'] = df['signal'].apply(analyze_sentiment)
    df.to_csv(output_file, index=False)
    print(f"Sentiment analysis saved to {output_file}")
