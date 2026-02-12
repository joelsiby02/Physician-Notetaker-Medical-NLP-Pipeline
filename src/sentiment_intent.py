import re
from typing import Dict, Any, List
from transformers import pipeline


def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def map_sentiment(label: str, score: float) -> str:
    """
    SST-2 labels: POSITIVE, NEGATIVE
    We map:
      NEGATIVE -> Anxious
      POSITIVE -> Reassured
    If score is low, call it Neutral.
    """
    if score < 0.70:
        return "Neutral"
    if label.upper() == "NEGATIVE":
        return "Anxious"
    return "Reassured"


def detect_intents(patient_text: str) -> List[str]:
    t = patient_text.lower()
    intents = []

    if re.search(r"\b(worried|concerned|need to worry|affect me)\b", t):
        intents.append("Seeking reassurance")

    if re.search(r"\b(pain|ache|hurt|stiff|discomfort)\b", t):
        intents.append("Reporting symptoms")

    if re.search(r"\b(thank you|appreciate)\b", t):
        intents.append("Expressing gratitude")

    if re.search(r"\b(do i|should i|can i)\b", t):
        intents.append("Asking a question")

    if not intents:
        intents.append("General conversation")

    return sorted(list(set(intents)))


def analyze_sentiment_and_intent(patient_text: str) -> Dict[str, Any]:
    model = load_sentiment_model()
    pred = model(patient_text[:1200])[0]  # keep it short for speed

    sentiment = map_sentiment(pred["label"], pred["score"])
    intents = detect_intents(patient_text)

    return {
        "Sentiment": sentiment,
        "Sentiment_Model": pred,
        "Intent": intents
    }
