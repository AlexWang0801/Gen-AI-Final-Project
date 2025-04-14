import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Fine-grained GoEmotions labels
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", 
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", 
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", 
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Mapping to 7-category labels
emotion_mapping = {
    "admiration": "joy", "amusement": "joy", "approval": "joy", "caring": "joy",
    "desire": "joy", "excitement": "joy", "gratitude": "joy", "joy": "joy",
    "love": "joy", "optimism": "joy", "pride": "joy", "relief": "joy",

    "anger": "anger", "annoyance": "anger",

    "disgust": "disgust", "disapproval": "disgust",

    "fear": "fear", "nervousness": "fear",

    "sadness": "sadness", "disappointment": "sadness", "grief": "sadness", "remorse": "sadness",

    "surprise": "surprise", "realization": "surprise",

    "neutral": "neutral", "confusion": "neutral", "curiosity": "neutral", "embarrassment": "neutral"
}

model_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Load sample
def load_goemotions():
    df = pd.read_csv('goemotions_1.csv')
    return df.sample(6666, random_state=42)

# Load bhadresh-savani model
emotion_model = pipeline(
    "text-classification",
    model="bhadresh-savani/distilbert-base-uncased-emotion",
    tokenizer="bhadresh-savani/distilbert-base-uncased-emotion",
    truncation=True
)

# Get model prediction
def get_huggingface_emotion(text):
    prediction = emotion_model(text)
    return prediction[0]["label"].lower()  # lowercase to match mapping

# Evaluate
def evaluate_models():
    test_data = load_goemotions()
    results = []

    for _, row in test_data.iterrows():
        text = row["text"]
        raw_true = row[emotion_labels].astype(int).to_dict()

        true_model_label = None
        for emo, val in raw_true.items():
            if val == 1 and emo in emotion_mapping:
                true_model_label = emotion_mapping[emo]
                break

        if true_model_label is None:
            continue

        pred_label = get_huggingface_emotion(text)
        results.append({
            "Text": text,
            "True Label": true_model_label,
            "Prediction": pred_label
        })

    return results

# Plot
def plot_results(results):
    correct = {label: 0 for label in model_labels}
    total = {label: 0 for label in model_labels}

    for r in results:
        true = r["True Label"]
        pred = r["Prediction"]
        if true in model_labels:
            total[true] += 1
            if pred == true:
                correct[true] += 1

    accuracy = [correct[label] / total[label] if total[label] > 0 else 0 for label in model_labels]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_labels, y=accuracy, palette="mako")
    plt.title("Accuracy per Emotion (bhadresh-savani/distilbert-base-uncased-emotion)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.show()

# Run
results = evaluate_models()
plot_results(results)
