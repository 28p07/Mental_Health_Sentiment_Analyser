import os
import sys
import pickle
import torch
import nltk
from flask import Flask, request, jsonify, render_template
from nltk.tokenize import word_tokenize
from src.logger import logging
from src.exception import CustomException
from src.components.data_tansformation import DataTranformation
from src.components.model_evaluator import LSTMModel

# Initialize Flask app
app = Flask(__name__)

# Download NLTK tokenizer if not already present
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Load word_to_index
try:
    with open("D:/Projects/mental_health_sentiment/artifacts/word_to_index.pkl", "rb") as f:
        word_to_index = pickle.load(f)
except Exception as e:
    raise CustomException(f"Error loading word_to_index.pkl: {e}", sys)

vocab_size = len(word_to_index)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load LSTM model
model = LSTMModel(vocab_size=vocab_size, embedding_dim=256, hidden_dim=64, output_dim=7).to(device)
try:
    model.load_state_dict(torch.load("D:/Projects/mental_health_sentiment/artifacts/model.pth", map_location=device))
except Exception as e:
    raise CustomException(f"Error loading model weights: {e}", sys)

model.eval()

# Sentiment mapping
class_map = {
    0: 'Normal',
    1: 'Depression',
    2: 'Suicidal',
    3: 'Anxiety',
    4: 'Bipolar',
    5: 'Tress',
    6: 'Personality disorder'
}

# Health check route
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Mental Health Sentiment API is running"}), 200

# Serve HTML form
@app.route("/form", methods=["GET"])
def form():
    return render_template("form.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        statement = data.get("text", "")

        if not statement:
            return jsonify({"error": "No input text provided"}), 400

        # Preprocess and encode
        data_transformation = DataTranformation()
        cleaned_text = data_transformation.clean(statement)
        tokenized_sentence = word_tokenize(cleaned_text)

        if not tokenized_sentence:
            return jsonify({"error": "Input statement is empty or only contains stopwords."}), 400

        encoded_input = data_transformation.encode_sentences(tokenized_sentence, word_to_index)
        input_tensor = torch.tensor(encoded_input, dtype=torch.long).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            output = model(input_tensor)
            predicted_class = torch.argmax(output, dim=1).item()

        return jsonify({
            "input": statement,
            "predicted_class": predicted_class,
            "sentiment": class_map[predicted_class]
        })

    except Exception as e:
        raise CustomException(f"Prediction failed: {e}", sys)

if __name__ == "__main__":
    app.run(debug=True)
