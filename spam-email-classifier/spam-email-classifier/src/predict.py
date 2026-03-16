import joblib
from src.preprocess import clean_text

def load_model_and_vectorizer(
    model_path='models/logistic_regression.pkl',
    vectorizer_path='models/vectorizer.pkl'
):
    """Load a saved model and vectorizer from disk."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_email(text, model, vectorizer):
    """Predict whether a single email is spam or ham."""
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]

    label = "🚨 SPAM" if prediction == 1 else "✅ HAM"
    confidence = probability[prediction] * 100

    print(f"\nEmail   : {text[:80]}{'...' if len(text) > 80 else ''}")
    print(f"Result  : {label}")
    print(f"Confidence: {confidence:.2f}%")
    return label, confidence
