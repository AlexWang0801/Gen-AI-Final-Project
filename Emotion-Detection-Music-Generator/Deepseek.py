import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns

# Set your OpenRouter API key here
API_KEY = "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"

emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", 
    "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", 
    "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism", 
    "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
]

def load_goemotions():
    df = pd.read_csv('goemotions_1.csv')  # Adjust path if needed
    return df.sample(6666, random_state=42)

test_data = load_goemotions()

def emotion_prompt(user_message):
    return f"What emotions are expressed in the following text? Reply with a list of emotions (separated by commas).\n\nText: \"{user_message}\""

def get_emotion(message):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    prompt = emotion_prompt(message)

    data = {
        "model": "deepseek/deepseek-r1:free", 
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
        print(f"Status Code: {response.status_code}")
        response.raise_for_status()
        emotion = response.json()["choices"][0]["message"]["content"]
        return emotion.strip()
    except:
        return ""

results = []

for sample in test_data.itertuples():
    true_labels = sample[10:]  # Adjust based on actual column index
    predicted_emotions = get_emotion(sample.text)
    predicted_labels = [1 if emotion in predicted_emotions.lower() else 0 for emotion in emotion_labels]

    results.append({
        "True Labels": true_labels,
        "Predicted Labels": predicted_labels
    })

correct_count = [0] * len(emotion_labels)
incorrect_count = [0] * len(emotion_labels)

for idx in range(len(emotion_labels)):
    for result in results:
        if result["True Labels"][idx] == result["Predicted Labels"][idx]:
            correct_count[idx] += 1
        else:
            incorrect_count[idx] += 1

accuracy = [correct - incorrect for correct, incorrect in zip(correct_count, incorrect_count)]

# Plot the accuracy chart
plt.figure(figsize=(12, 6))
sns.barplot(x=emotion_labels, y=accuracy, palette="Spectral")
plt.title("Emotion Detection Accuracy (Correct - Incorrect)")
plt.xlabel("Emotion")
plt.ylabel("Net Accuracy")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
