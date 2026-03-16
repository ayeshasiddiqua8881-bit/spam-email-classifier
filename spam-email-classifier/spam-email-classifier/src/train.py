import joblib
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def train_naive_bayes(X_train, y_train, save_path='models/naive_bayes.pkl'):
    """Train and save a Multinomial Naive Bayes model."""
    model = MultinomialNB()
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[✓] Naive Bayes model saved to {save_path}")
    return model

def train_logistic_regression(X_train, y_train, save_path='models/logistic_regression.pkl'):
    """Train and save a Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"[✓] Logistic Regression model saved to {save_path}")
    return model
