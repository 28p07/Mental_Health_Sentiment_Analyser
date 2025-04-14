import os
import sys
import re
import pickle
import torch
import nltk
from nltk.tokenize import word_tokenize
from src.logger import logging
from src.exception import CustomException
from src.components.data_tansformation import DataTranformation
from src.components.model_evaluator import LSTMModel

# Download NLTK tokenizer
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Load word_to_index dictionary
try:
    with open("D:/Projects/mental_health_sentiment/artifacts/word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)
except Exception as e:
    raise CustomException(f"Error loading word_to_index.pkl: {e}", sys)

vocab_size = len(word_to_index)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = LSTMModel(vocab_size=vocab_size, embedding_dim=256, hidden_dim=64, output_dim=7).to(device)
try:
    model.load_state_dict(torch.load("D:/Projects/mental_health_sentiment/artifacts/model.pth", map_location=device))
except Exception as e:
    raise CustomException(f"Error loading model weights: {e}", sys)

model.eval()  # Set model to evaluation mode

# Take user input
statement = input("Enter the statement: ")

# Preprocess input
data_transformation = DataTranformation()
cleaned_text = data_transformation.clean(statement)
tokenized_sentence = word_tokenize(cleaned_text)

# Handle empty input
if not tokenized_sentence:
    print("Input statement is empty or only contains stopwords.")
    sys.exit()

# Encode input sentence
embedding = data_transformation.encode_sentences(tokenized_sentence, word_to_index)
embedding = torch.tensor(embedding, dtype=torch.long).unsqueeze(0).to(device)

# Make prediction
with torch.no_grad():
    output = model(embedding)
    predicted_class = torch.argmax(output, dim=1).item()


d = {0:'Normal',1:'Depression',2:'Suicidal',3:'Anxiety',4:'Bipolar',5:'Tress',6:'Personality disorder'}

print(f"Predicted Sentiment Class: {d[predicted_class]}")
