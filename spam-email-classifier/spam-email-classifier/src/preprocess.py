import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

def clean_text(text):
    """Lowercase, remove punctuation, digits, and extra whitespace."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_prepare(filepath):
    """Load dataset and return cleaned train/test splits."""
    df = pd.read_csv(filepath, encoding='latin-1')

    # Support both 'label'/'text' and 'v1'/'v2' column naming (SMS Spam dataset)
    if 'v1' in df.columns:
        df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    else:
        df = df[['label', 'text']]

    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].apply(clean_text)
    df.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    return X_train, X_test, y_train, y_test

def vectorize(X_train, X_test, save_path='models/vectorizer.pkl'):
    """Fit TF-IDF on training data and transform both splits."""
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(vectorizer, save_path)
    print(f"[✓] Vectorizer saved to {save_path}")
    return X_train_vec, X_test_vec, vectorizer
