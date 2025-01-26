import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import numpy as np

# Hugging Face model repository
MODEL_REPO = "karthikeyan6864/SentimentAppModel"
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_REPO)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_REPO)


# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1).detach().numpy()[0]
    labels = ["Positive", "Neutral", "Negative"]
    predicted_label = labels[np.argmax(probabilities)]
    confidence = np.max(probabilities)
    return predicted_label, confidence

# Streamlit app layout
st.title("Sentiment Analysis with Fine-Tuned DistilBERT")
st.markdown("Analyze the sentiment of your text using a fine-tuned DistilBERT model.")

# Input text
input_text = st.text_area("Enter your text below:")

if st.button("Predict Sentiment"):
    if input_text.strip():
        with st.spinner("Predicting sentiment..."):
            label, confidence = predict_sentiment(input_text)
            st.write(f"**Predicted Sentiment:** {label}")
            st.write(f"**Confidence:** {confidence:.2f}")
    else:
        st.warning("Please enter some text to analyze.")
