"""
Spam Email Classifier — Main Entry Point
Run: python main.py
"""

from src.preprocess import load_and_prepare, vectorize
from src.train import train_naive_bayes, train_logistic_regression
from src.evaluate import evaluate_model
from src.predict import load_model_and_vectorizer, predict_email

DATASET_PATH = 'data/spam.csv'

def main():
    print("\n📧 Spam Email Classifier")
    print("=" * 45)

    # 1. Load & preprocess
    print("\n[1/4] Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_prepare(DATASET_PATH)
    print(f"      Train samples : {len(X_train)}")
    print(f"      Test samples  : {len(X_test)}")

    # 2. Vectorize
    print("\n[2/4] Vectorizing text with TF-IDF...")
    X_train_vec, X_test_vec, _ = vectorize(X_train, X_test)

    # 3. Train both models
    print("\n[3/4] Training models...")
    nb_model = train_naive_bayes(X_train_vec, y_train)
    lr_model = train_logistic_regression(X_train_vec, y_train)

    # 4. Evaluate both models
    print("\n[4/4] Evaluating models...")
    evaluate_model(nb_model, X_test_vec, y_test, "Naive Bayes")
    evaluate_model(lr_model, X_test_vec, y_test, "Logistic Regression")

    # 5. Demo predictions
    print("\n" + "=" * 45)
    print("  DEMO — Predicting on sample emails")
    print("=" * 45)

    model, vectorizer = load_model_and_vectorizer()

    sample_emails = [
        "Congratulations! You've won a $1000 Walmart gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow at 1pm?",
        "URGENT: Your account has been suspended. Verify your details immediately.",
        "Please find the project report attached as discussed in our last meeting."
    ]

    for email in sample_emails:
        predict_email(email, model, vectorizer)

if __name__ == "__main__":
    main()
