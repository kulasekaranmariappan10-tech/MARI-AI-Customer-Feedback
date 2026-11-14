# 🤖 MARI AI – Customer Feedback Analyzer

An NLP-based Customer Feedback Analyzer that uses Machine Learning to classify customer reviews into Positive, Negative, and Neutral sentiments.

## 🚀 Project Overview

MARI AI analyzes customer feedback and predicts the sentiment behind each review.

It uses Natural Language Processing (NLP), TF-IDF Vectorization, and the Naive Bayes algorithm to perform sentiment classification.

The application provides an interactive web interface built using Streamlit.

## ✨ Features

- Customer Review Analysis
- Positive, Negative, and Neutral Sentiment Classification
- Text Preprocessing
- TF-IDF Vectorization
- Naive Bayes Machine Learning Model
- Interactive Streamlit Web Interface
- Real-time Sentiment Prediction

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- NLP
- TF-IDF
- Multinomial Naive Bayes
- Streamlit
- Joblib

## 📂 Project Structure

```text
MARI-AI/
│
├── data/
│   └── reviews.csv
│
├── app.py
├── train.py
├── requirements.txt
├── sentiment_model.pkl
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to Project

```bash
cd MARI-AI
```

### 3. Create Virtual Environment

```bash
python -m venv venv
```

### 4. Activate Environment

Windows:

```powershell
.\venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 🧠 Train the Model

Run:

```bash
python train.py
```

The model will be trained using the customer review dataset and saved as:

```text
sentiment_model.pkl
```

## ▶️ Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

## 💡 Example

**Input:**

```text
The hotel food is amazing.
```

**Output:**

```text
Sentiment: Positive
```

**Input:**

```text
The room is very dirty.
```

**Output:**

```text
Sentiment: Negative
```

## 📊 Dataset

The project uses a sample customer review dataset containing:

- Customer Reviews
- Sentiment Labels
- Review Categories

The current dataset is small and intended for learning and demonstration.

## 🎯 Future Improvements

- Sentiment Confidence Scores
- Complaint Category Classification
- Customer Feedback Dashboard
- Review Analytics
- Larger Real-World Dataset
- Database Integration
- Advanced Transformer-Based NLP Models

## 👨‍💻 Author

**Mariappan K
