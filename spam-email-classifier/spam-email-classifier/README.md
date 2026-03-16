# 📧 Spam Email Classifier

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A machine learning project that classifies emails/SMS messages as **spam or ham (not spam)** using Natural Language Processing (NLP) techniques.

---

## 📌 Project Overview

Spam detection is a classic and practical NLP problem. This project builds and compares two ML models — **Naive Bayes** and **Logistic Regression** — using TF-IDF features extracted from raw text data.

**Key highlights:**
- Full text preprocessing pipeline (cleaning, normalization)
- TF-IDF vectorization with bigrams
- Side-by-side model comparison with metrics
- Confusion matrix visualization
- Reusable predict function for new inputs
- Saved model with pickle for deployment

---

## 📁 Project Structure

```
spam-email-classifier/
│
├── data/
│   ├── spam.csv                  # Raw dataset (download link below)
│   ├── class_distribution.png   # EDA visualization
│   └── confusion_matrices.png   # Model evaluation plots
│
├── models/
│   ├── spam_classifier.pkl      # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl     # Fitted TF-IDF vectorizer
│
├── notebooks/
│   └── spam_classifier.ipynb   # Main notebook (full pipeline)
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

- **Source:** [SMS Spam Collection Dataset — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 messages
- **Classes:** `ham` (87%) and `spam` (13%)

> Download the dataset and place it in the `data/` folder as `spam.csv`.

---

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/spam-email-classifier.git
cd spam-email-classifier

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook notebooks/spam_classifier.ipynb
```

---

## 🧠 ML Pipeline

```
Raw Text
   ↓
Preprocessing (lowercase, remove URLs, numbers, punctuation)
   ↓
TF-IDF Vectorization (5000 features, unigrams + bigrams)
   ↓
Model Training (Naive Bayes + Logistic Regression)
   ↓
Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
   ↓
Save Best Model (pickle)
```

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | ~97% | ~97% | ~93% | ~95% |
| **Logistic Regression** | **~98%** | **~98%** | **~95%** | **~97%** |

> ✅ **Logistic Regression** is the best performing model.

---

## 🔍 Sample Predictions

```
Input : "Congratulations! You've won a free iPhone. Click here to claim now!"
Output: 🚨 SPAM (98.5% confidence)

Input : "Hey, are we still meeting tomorrow at 10am?"
Output: ✅ HAM (99.2% confidence)
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| pandas / numpy | Data handling |
| scikit-learn | ML models & TF-IDF |
| matplotlib / seaborn | Visualization |
| Jupyter Notebook | Development environment |
| pickle | Model serialization |

---

## 🚀 Future Improvements

- [ ] Add deep learning model (LSTM / BERT)
- [ ] Build a simple web UI with Streamlit
- [ ] Expand to multi-class email categorization
- [ ] Deploy as REST API using FastAPI

---

## 👤 Author

**Aladdin**  
B.Sc. IT Student | Aspiring AI Engineer  
📍 Maharashtra, India  
🔗 [LinkedIn](https://linkedin.com) | [GitHub](https://github.com)

---

## 📄 License

This project is licensed under the MIT License.
