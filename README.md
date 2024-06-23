# Israel_Palestine_war_SentimentAnalysis_RedditComments
Sentimental Analysis of Reddit Comments on Israel-Palestine War using VADER
# Sentiment Analysis of Israel-Palestine Reddit Comments

This project performs sentiment analysis on Reddit comments related to the Israel-Palestine conflict. It utilizes various NLP techniques including TF-IDF vectorization, K-Means clustering, VADER sentiment analysis, and Latent Dirichlet Allocation (LDA) for topic modeling.

## Table of Contents
- [Installation]
- [Data Preparation]
- [Sentiment Analysis Using VADER]
- [Exploratory Data Analysis]
- [Positive or Negative Bias Detection]
- [Topic Modelling]
- [Geopolitical Stance Determination]
- [Visualization]

## Installation

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

3. Ensure NLTK corpora are downloaded:
    ```python
    import nltk
    nltk.download('vader_lexicon')
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    ```

## Data Preparation

The dataset used in this project is a CSV file containing Reddit comments. The data is preprocessed to remove HTML tags, URLs, and non-alphanumeric characters, and then tokenized and lemmatized.

## Sentiment Analysis Using VADER

We use the VADER sentiment analysis tool to generate sentiment scores for each comment, including compound, positive, negative, and neutral scores.

## Exploratory Data Analysis

We explore the distribution of sentiment categories and visualize the results using pie charts.

## Positive or Negative Bias Detection

We analyze subreddit sentiment scores to detect biases and determine whether the overall sentiment is positive, negative, or neutral.

## Topic Modelling

Using Latent Dirichlet Allocation (LDA), we identify the main topics discussed in the comments and determine their associations with the sentiments expressed.

## Geopolitical Stance Determination

We determine the geopolitical stance of comments based on their sentiment scores and dominant topics, categorizing them as supportive or against Israel/Palestine, or neutral.


