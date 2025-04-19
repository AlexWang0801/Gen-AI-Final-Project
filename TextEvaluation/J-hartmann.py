import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Fine-grained GoEmotions labels (columns from CSV)
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", 
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", 
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", 
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

# Mapping fine-grained GoEmotions to j-hartmann 7 labels
emotion_mapping = {
    "admiration": "joy", "amusement": "joy", "approval": "joy", "caring": "joy",
    "desire": "joy", "excitement": "joy", "gratitude": "joy", "joy": "joy",
    "love": "joy", "optimism": "joy", "pride": "joy", "relief": "joy",

    "anger": "anger", "annoyance": "anger",

    "disgust": "disgust", "disapproval": "disgust",

    "fear": "fear", "nervousness": "fear",

    "sadness": "sadness", "disappointment": "sadness", "grief": "sadness", "remorse": "sadness",

    "surprise": "surprise", "realization": "surprise",

    "neutral": "neutral",

    "confusion": "neutral", "curiosity": "neutral", "embarrassment": "neutral"
}

model_labels = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]

# Load sample from GoEmotions
def load_goemotions():
    df = pd.read_csv('goemotions_1.csv')
    return df.sample(6666, random_state=42)

# Load model
emotion_model = pipeline(
    "text-classification", 
    model="j-hartmann/emotion-english-distilroberta-base", 
    tokenizer="j-hartmann/emotion-english-distilroberta-base",
    truncation=True
)

# Predict single label
def get_huggingface_emotion(text):
    prediction = emotion_model(text)
    return prediction[0]["label"]

# Evaluation
def evaluate_models():
    test_data = load_goemotions()
    results = []

    for _, row in test_data.iterrows():
        text = row["text"]
        raw_true = row[emotion_labels].astype(int).to_dict()

        # Map first matching emotion to model emotion
        true_model_label = None
        for emo, val in raw_true.items():
            if val == 1 and emo in emotion_mapping:
                true_model_label = emotion_mapping[emo]
                break

        if true_model_label is None:
            continue  # Skip if no mappable label

        pred_label = get_huggingface_emotion(text)
        results.append({
            "Text": text,
            "True Label": true_model_label,
            "Prediction": pred_label
        })

    return results

# Plot results
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

    # Calculate accuracy
    accuracy = [correct[label] / total[label] if total[label] > 0 else 0 for label in model_labels]

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=model_labels, y=accuracy, palette="mako")
    plt.title("Accuracy per Emotion (Mapped to j-hartmann categories)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.show()

# Run everything
results = evaluate_models()
plot_results(results)
