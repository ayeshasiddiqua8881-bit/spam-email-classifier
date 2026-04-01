# 📧 Spam Email Classifier

A machine learning project that classifies emails as **spam** or **ham (not spam)** using Natural Language Processing techniques. Built with Python, scikit-learn, and TF-IDF vectorization.

---

## 🚀 Demo

```
Email   : Congratulations! You've won a $1000 Walmart gift card. Click here...
Result  : 🚨 SPAM
Confidence: 98.73%

Email   : Hey, are we still meeting for lunch tomorrow at 1pm?
Result  : ✅ HAM
Confidence: 99.21%
```

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Naive Bayes | 97.8% | 96.4% | 93.2% | 94.8% |
| Logistic Regression | 98.4% | 97.1% | 95.6% | 96.3% |

> Logistic Regression is used as the default model for predictions.

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **ML Library:** scikit-learn
- **NLP:** TF-IDF Vectorization
- **Models:** Multinomial Naive Bayes, Logistic Regression
- **Data Handling:** pandas

---

## 📁 Project Structure

```
spam-email-classifier/
│
├── data/
│   └── spam.csv               # Dataset (SMS Spam Collection)
│
├── models/
│   ├── naive_bayes.pkl        # Saved Naive Bayes model
│   ├── logistic_regression.pkl
│   └── vectorizer.pkl         # Saved TF-IDF vectorizer
│
├── src/
│   ├── preprocess.py          # Text cleaning & TF-IDF vectorization
│   ├── train.py               # Model training
│   ├── evaluate.py            # Metrics: Accuracy, Precision, Recall, F1
│   └── predict.py             # Predict on new emails
│
├── main.py                    # Entry point — runs full pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/your-username/spam-email-classifier.git
cd spam-email-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle and place `spam.csv` inside the `data/` folder.

### 4. Run the full pipeline
```bash
python main.py
```

This will preprocess the data, train both models, print evaluation metrics, and run demo predictions.

---

## 🧠 How It Works

1. **Text Preprocessing** — Emails are lowercased, stripped of punctuation and digits, and cleaned of extra whitespace.
2. **TF-IDF Vectorization** — Converts text into numerical features based on word frequency and importance (top 5000 features).
3. **Model Training** — Two models are trained: Multinomial Naive Bayes and Logistic Regression.
4. **Evaluation** — Models are evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
5. **Prediction** — The saved model and vectorizer are loaded to classify new email inputs.

---

## 📈 Why These Models?

- **Naive Bayes** is a classic baseline for text classification — fast, simple, and surprisingly effective for spam detection.
- **Logistic Regression** generally achieves higher precision and recall, making it the better choice when minimizing false positives matters.

---

## 📬 Dataset

[SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) — 5,572 labeled SMS messages (ham/spam).

---

## 👤 Author

**Ayesha Siddiqua**
B.Sc. IT Student | Aspiring AI Engineer
🔗 [LinkedIn](https://www.linkedin.com/in/ayeshasiddiquaabdulkalam71ab7a391/) 

---

## 📄 License

This project is open source under the [MIT License](LICENSE).
