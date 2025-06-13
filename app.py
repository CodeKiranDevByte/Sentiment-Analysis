# app.py
import streamlit as st
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from wordcloud import WordCloud
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('vader_lexicon')

# Upload CSV file
st.title("Sentiment Analysis App")
uploaded_file = st.file_uploader("Upload a CSV file with reviewText column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # Basic cleaning
    df["reviewText"] = df["reviewText"].astype(str)
    sid = SentimentIntensityAnalyzer()

    # Sentiment analysis with VADER
    df["polarity"] = df["reviewText"].apply(lambda x: TextBlob(x).sentiment.polarity)
    df["subjectivity"] = df["reviewText"].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    df["sentiment"] = df["polarity"].apply(lambda x: "positive" if x > 0 else "negative" if x < 0 else "neutral")

    # Pie chart of sentiments
    st.subheader("Sentiment Distribution")
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Countplot', 'Pie Chart'),
                        specs=[[{"type": "xy"}, {"type": "domain"}]])

    constraints = ['#834D22', '#EBE00C', '#1FEB0C', '#0C92EB', '#EB0CD5']
    fig.add_trace(go.Bar(y=df["sentiment"].value_counts().values.tolist(),
                         x=df["sentiment"].value_counts().index,
                         text=df["sentiment"].value_counts().values,
                         textposition='auto',
                         marker=dict(color=constraints)), row=1, col=1)

    fig.add_trace(go.Pie(labels=df["sentiment"].value_counts().index,
                         values=df["sentiment"].value_counts().values,
                         marker=dict(colors=constraints)), row=1, col=2)
    
    st.plotly_chart(fig)

    # WordCloud for positive reviews
    st.subheader("Word Cloud for Positive Reviews")
    positive_text = " ".join(df[df["sentiment"] == "positive"]["reviewText"])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
    st.image(wordcloud.to_array())

    # Show top reviews
    st.subheader("Top Positive Reviews")
    st.dataframe(df[df["sentiment"] == "positive"].head(5)[["reviewerName", "reviewText"]])
