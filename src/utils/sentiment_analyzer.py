import os
import feedparser
import requests
import numpy as np
from datetime import timedelta
from transformers import pipeline
from loguru import logger
import tweepy
from src.utils import DataCache

class SentimentAnalyzer:
    """Sentiment analysis for stock market using Google News RSS + Twitter"""

    def __init__(self, cache_minutes=10):
        self.sentiment_pipeline = pipeline("sentiment-analysis")    # uses DistilBERT base uncased finetuned SST-2
        self.cache = DataCache(cache_dir="sentiment_cache", max_age_hours=cache_minutes / 60)

        # Twitter setup
        self.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if self.twitter_bearer_token:
            self.twitter_client = tweepy.Client(bearer_token=self.twitter_bearer_token)
        else:
            self.twitter_client = None
            logger.warning("TWITTER_BEARER_TOKEN not set. Social sentiment will be skipped.")

    def _analyze_text_list(self, texts):
        """Run sentiment on a list of texts and return average score"""
        sentiments = []
        for text in texts:
            text = text.strip()
            if not text:
                continue
            try:
                result = self.sentiment_pipeline(text[:512])[0]  # limit text length
                score = result['score'] if result['label'] == 'POSITIVE' else -result['score']
                sentiments.append(score)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
        return np.mean(sentiments) if sentiments else 0.0

    # TODO:: custom scrape for better results.
    def analyze_news_sentiment(self, symbol):
        """Pull finance news from Google News RSS and analyze sentiment"""
        rss_urls = [
            f"https://news.google.com/rss/search?q={symbol}+site:moneycontrol.com&hl=en-IN&gl=IN&ceid=IN:en",
            f"https://news.google.com/rss/search?q={symbol}+site:economictimes.indiatimes.com&hl=en-IN&gl=IN&ceid=IN:en",
            f"https://news.google.com/rss/search?q={symbol}+site:livemint.com&hl=en-IN&gl=IN&ceid=IN:en"
        ]
        articles = []
        for url in rss_urls:
            feed = feedparser.parse(url)
            for entry in feed.entries[:5]:  # limit per source
                articles.append(entry.title + " " + getattr(entry, "summary", ""))

        return self._analyze_text_list(articles)

    def analyze_social_sentiment(self, symbol):
        """Pull tweets with hashtags and analyze sentiment"""
        if not self.twitter_client:
            return 0.0  # fallback

        hashtags = [symbol, f"#{symbol}", "#NSE", "#BSE"]
        query = " OR ".join(hashtags) + " lang:en -is:retweet"

        try:
            tweets = self.twitter_client.search_recent_tweets(
                query=query,
                max_results=20,
                tweet_fields=["text"]
            )
            tweet_texts = [tweet.text for tweet in tweets.data] if tweets.data else []
            return self._analyze_text_list(tweet_texts)
        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")
            return 0.0

    def get_combined_sentiment(self, symbol, debug=False):
        """Get sentiment from cache or compute fresh"""
        cache_key = f"sentiment_{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            if debug:
                print(f"[DEBUG] Cache hit for {symbol}: {cached}")
            return cached

        news_score = self.analyze_news_sentiment(symbol)
        social_score = self.analyze_social_sentiment(symbol)

        combined = 0.7 * news_score + 0.3 * social_score
        combined = max(-1, min(1, combined))

        if debug:
            print(f"[DEBUG] News: {news_score}, Social: {social_score}, Combined: {combined}")

        self.cache.set(cache_key, combined)
        return combined