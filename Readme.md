# Sentiment Analysis with Fine-Tuned DistilBERT

Analyze the sentiment of text (Positive, Neutral, Negative) using a fine-tuned **DistilBERT** model. This project includes training the model on Twitter data, deploying it on Hugging Face, and creating a **Streamlit app** for real-time sentiment predictions.

---

##  Features

- Fine-tuned **DistilBERT** model for sentiment classification.
- Preprocessing pipeline for cleaning and tokenizing text data.
- Streamlit app for interactive, real-time sentiment predictions.
- Deployed model on Hugging Face for public use and integration.

---

##  Dataset

The dataset used in this project is a Twitter sentiment dataset with the following columns:
- `Tweet_Id`: Unique ID for each tweet.
- `Entity`: Topic or subject of the tweet.
- `label`: Sentiment label (Positive, Neutral, Negative).
- `text`: The actual text content of the tweet.

### Dataset Cleaning Steps:
1. Irrelevant labels were replaced with Neutral.
2. Removed:
   - URLs (e.g., `http://...`, `www...`).
   - Mentions (e.g., `@username`).
   - Hashtags (e.g., `#topic`).
3. Converted text to lowercase and removed extra spaces.

## Data Preprocessing
The dataset used is a Twitter sentiment dataset with the following columns:
   -Tweet_Id: Unique identifier for tweets.
   -Entity: Associated entity for the tweet.
   -label: Sentiment label (Positive, Neutral, Negative, or Irrelevant).
   -text: The tweet text.
### Preprocessing steps:
Mapped the Irrelevant label to Neutral and encoded labels into numeric values:
 -Positive → 0
 -Neutral → 1
 -Negative → 2
Removed URLs, hashtags, mentions, and special characters from the text.
Converted text to lowercase and stripped extra whitespace.
## Model Training
Used Hugging Face Transformers for model training.
Tokenized the text using DistilBertTokenizer.
Fine-tuned DistilBertForSequenceClassification with:
Training Arguments:
batch_size: 16
epochs: 3
## evaluation_strategy: Evaluate at the end of each epoch.
## Evaluation Metric:
## Accuracy
F1-Score (weighted)
## Deployment
The trained model and tokenizer were saved locally and uploaded to:
Google Drive for backup.
Hugging Face Hub for public deployment.
## Streamlit App
A user-friendly app allows users to:
Input custom text.
View predicted sentiment and confidence score.

## Access Online
The model is deployed on Hugging Face Spaces. You can access the web app directly using this link:
Sentiment Analysis App on Hugging Face Spaces
https://huggingface.co/spaces/karthikeyan6864/Sentimentapp

## Dataset
The dataset used for training was obtained from Twitter Training Dataset and includes labeled tweets for sentiment analysis.

## Model Performance
Metrics on Validation Set:
Accuracy: ~90%
F1-Score (Weighted): ~89%
Evaluation was based on a balanced test set with diverse tweets.

## Links
hugging face space: https://huggingface.co/spaces/karthikeyan6864/Sentimentapp/tree/main
hugging face model: https://huggingface.co/karthikeyan6864/SentimentAppModel/tree/main

##Acknowledgements
Hugging Face for providing excellent resources for NLP.
Streamlit for enabling easy deployment of interactive web apps.
Twitter Sentiment Dataset for training and evaluation.
