# Sentiment Analysis — Customer Complaint Triage

This project analyzes and classifies customer tweets directed at airlines as
Positive, Neutral, or Negative, using classical machine learning and NLP
techniques. The goal is to identify negative, high-urgency messages that
should be prioritized for customer support — a complaint triage use case.

## 📂 Project Structure

```
├── Sentiment_Analysis_Complaint_Triage.ipynb   # Full analysis: preprocessing, modeling, evaluation
├── Tweets.csv                                   # Dataset (see download note below)
└── README.md
```

**Note:** the dataset is not included in this repository due to file size.
Download it from Kaggle:
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
After downloading, place `Tweets.csv` in the project folder before running
the notebook.

## About the Dataset

Real tweets directed at 6 major US airlines, labeled by sentiment
(Twitter US Airline Sentiment, CrowdFlower). ~14,600 tweets total.

**Target variable:**
- Negative
- Neutral
- Positive

Class distribution is imbalanced — negative tweets substantially outnumber
neutral and positive ones, which reflects real-world behavior: customers
tweet at airlines far more often to complain than to praise.

## 🔑 Key Steps in the Notebook

**1. Data Exploration**
- Class distribution and severity of imbalance
- Sentiment breakdown by airline
- Most frequent words per sentiment class

**2. Text Preprocessing**
- Removing @mentions, URLs, and non-alphabetic characters
- Lowercasing and stopword removal
- Sanity-checking that cleaning didn't empty out too many tweets

**3. Feature Engineering**
- TF-IDF vectorization (unigrams + bigrams, top 5,000 features)
- Fit only on the training set to avoid leaking test-set vocabulary statistics

**4. Modeling & Evaluation**
- Logistic Regression, Multinomial Naive Bayes, Linear SVM — compared on
  Accuracy, Macro-F1, and Weighted-F1 (macro-F1 prioritized, since it doesn't
  let the majority negative class hide poor performance on neutral/positive)
- Confusion matrices across all three models to identify where
  misclassifications actually happen (typically around the neutral class,
  the hardest to define both for the model and the original human labelers)

**5. Interpretability**
- Logistic Regression coefficients used to identify the words most
  associated with each sentiment class — a simple, transparent alternative
  to SHAP that's well suited to linear models on TF-IDF features

**6. Business Framing — Complaint Triage Simulation**
- Tweets predicted negative are flagged as high priority for support routing
- Recall on the negative class is reported specifically, since missing an
  actual complaint (false negative) is more costly than over-flagging a
  neutral message

## 🚀 How to Run

```
pip install pandas numpy matplotlib seaborn scikit-learn nltk
jupyter notebook Sentiment_Analysis_Complaint_Triage.ipynb
```

## 📌 Requirements

- Python 3.8+
- Jupyter Notebook
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- NLTK (for stopword removal)

## 📈 Future Improvements

- Compare against a pretrained transformer (e.g. DistilBERT) fine-tuned on
  the same data, as a classical vs. modern NLP comparison
- Add aspect-based sentiment (e.g. separately flagging delay, baggage, or
  customer service complaints) instead of one overall sentiment label
- Deploy as a small web app for live message triage
