# 🧠 Mental Health Sentiment Analyser

A machine learning-based web application designed to detect and analyze sentiments from text data related to **mental health**. This project leverages NLP techniques and deep learning (LSTM) to classify input text as **Normal**, **Depression**,**Suicidal**,**Anxiety**,**Bipolar**,**Stress** or **Personality Disorder**.

---

## 📌 Key Features

- 🧾 Text sentiment classification using deep learning (LSTM)
- 🌐 Web interface for easy user interaction (Flask backend)
- 📊 Modular and scalable pipeline for data processing and prediction
- 📁 Well-structured project using custom pipelines and components
- 🧪 Experimentation notebooks included for model development and tuning

---

## 🗂️ Project Structure

```
Mental_Health_Sentiment_Analyser/
│
├── artifacts/                     # Contains model artifacts and datasets
│   ├── data.csv
│   ├── model.pth
│   ├── test.csv
│   ├── train.csv
│   ├── word_to_index.pkl
│
├── logs/                          # Logging and debug information
│
├── notebook/                      # Jupyter notebooks for experimentation
│   ├── data.csv
│   ├── experiments.ipynb
│   └── lstm.pth
│
├── src/                           # Source code directory
│   ├── components/                # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluator.py
│   │
│   ├── pipeline/                  # Pipeline orchestration modules
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   │   ├── prediction_pipeline.py
│   │   └── evaluation_pipeline.py
│   │
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── templates/                     # HTML templates for frontend
│   └── form.html
│
├── app.py                         # Flask application entry point
├── requirements.txt               # Python dependencies
├── setup.py                       # Project setup script
└── README.md                      # Project documentation
```

---

## 🚀 Getting Started

### 📦 Prerequisites

Ensure you have the following installed:

- Python ≥ 3.7
- pip (Python package manager)

### 🔧 Installation

1. **Clone the repository**  
```bash
git clone https://github.com/28p07/Mental_Health_Sentiment_Analyser.git
cd Mental_Health_Sentiment_Analyser
```

2. **Install dependencies**  
```bash
pip install -r requirements.txt
```

3. **Run the application**  
```bash
python app.py
```

Then open your browser at `http://127.0.0.1:5000` to access the web app.

---

## 💡 How It Works

1. The user enters text via a simple web form.
2. The text is preprocessed and tokenized.
3. A pretrained LSTM model makes the prediction.
4. The sentiment result (Positive, Neutral, Negative) is returned to the frontend.

---

## 📊 Model Details

- Model Type: **LSTM (Long Short-Term Memory)**
- Frameworks: **PyTorch, scikit-learn**
- Evaluation Metrics: **Accuracy, Confusion Matrix**
- Trained using a labeled mental health dataset for sentiment classification.

---

## 🔬 Notebooks

Explore the `notebook/experiments.ipynb` to see:

- Data preprocessing
- Tokenization & sequence padding
- Model architecture definition
- Training & validation curves
- Model saving and loading

---

## 🧪 Pipelines & Components

All ML logic is modularized:

| Component             | Description                                      |
|----------------------|--------------------------------------------------|
| `data_ingestion.py`  | Loads and splits data into training/testing sets |
| `data_transformation.py` | Text cleaning, tokenization, and vectorization |
| `model_trainer.py`   | Builds and trains the LSTM model                 |
| `model_evaluator.py` | Evaluates predictions using metrics              |
| `predict_pipeline.py`| Performs end-to-end prediction on new input      |

---

## 🖼️ Frontend Preview

A simple web interface created using Flask + HTML.

- `form.html` allows users to input text.
- Sentiment result is dynamically rendered.

---

## 📄 License

This project is licensed under the **MIT License**.

---


## 🌟 Acknowledgements

Thanks to the open-source community and contributors who support NLP research and mental health awareness.
