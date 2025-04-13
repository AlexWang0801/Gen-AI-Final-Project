import streamlit as st
import os
import random
import json
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# --- Page config must come first ---
st.set_page_config(page_title="🎵 Emotion Music Generator (MusicGen)", layout="centered")

# --- Load emotion labels ---
with open("data/emotions.txt", "r") as f:
    emotions = [line.strip() for line in f]

# --- Load per-class optimal thresholds ---
with open("src/optimal_thresholds.json") as f:
    thresholds = json.load(f)

# --- Manual mapping to MusicGen prompts ---
manual_map = {
    "anger": "tense",
    "annoyance": "tense",
    "fear": "tense",
    "grief": "sad",
    "sadness": "sad",
    "joy": "happy",
    "gratitude": "happy",
    "relief": "peaceful"
}

# --- Description ---
emotion_descriptions = {
    "joy": "You're feeling joyful and cheerful! 😊",
    "sadness": "Sounds like you're feeling a bit down. 😢",
    "anger": "You're expressing anger or frustration. 😠",
    "gratitude": "You're feeling thankful and appreciative. 🙏",
    "love": "Your message shows love and affection. ❤️",
    "fear": "There’s a sense of fear or anxiety. 😨",
    "surprise": "That caught you off guard! 😲",
    "neutral": "A calm, neutral mood. 😐",
    "peaceful": "You're calm and at ease. 🌿",
    "tense": "There’s tension or unease. 😬",
}

@st.cache_resource
def load_emotion_model():
    model="bhadresh-savani/distilbert-base-uncased-emotion"
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=model,
        top_k=None,
        function_to_apply="sigmoid"
    )

emotion_classifier = load_emotion_model()

# --- MusicGen ---
@st.cache_resource
def load_musicgen_model():
    return MusicGen.get_pretrained('facebook/musicgen-small')

musicgen = load_musicgen_model()

# --- Embedding for emotion similarity ---
embedder = SentenceTransformer('all-mpnet-base-v2')
music_labels = ["happy", "tense", "sad", "peaceful"]
music_embs = embedder.encode(music_labels, convert_to_tensor=True)

def map_emotion_to_music_auto(predicted_label: str) -> str:
    label = predicted_label.lower()
    if label in manual_map:
        return manual_map[label]
    emo_emb = embedder.encode(label, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emo_emb, music_embs)[0]
    best_idx = int(torch.argmax(cosine_scores))
    return music_labels[best_idx]

# --- UI ---
st.title("🎵 Emotion-to-Music Generator (with MusicGen)")
st.markdown("""
Enter a sentence describing how you feel, and this app will:
1. Detect your emotion using a fine-tuned BERT model 🤖  
2. Generate a piece of **audio** music with Meta's MusicGen 🎶  
3. Let you listen to and download your emotional melody 🎧  
""")

user_input = st.text_area("💬 What are you feeling?", placeholder="e.g. I feel calm and peaceful watching the rain fall...")
duration = st.slider("🎛️ Music Duration (seconds)", min_value=5, max_value=30, value=10, step=5)

if user_input:
    with st.spinner("🔍 Detecting emotion..."):
        raw_scores = emotion_classifier(user_input)[0]
        predicted_emotions = []
        for item in raw_scores:
            label = item["label"]
            score = item["score"]
            if score > thresholds.get(label, 0.5):
                predicted_emotions.append({"label": label, "score": score})
        top_emotions = sorted(predicted_emotions, key=lambda x: x["score"], reverse=True)[:2]

    st.subheader("🎯 Detected Emotions:")
    for emo in top_emotions:
        label = emo["label"]
        st.markdown(f"**• {label.capitalize()} ({emo['score']:.2f})**")
        if label in emotion_descriptions:
            st.caption(emotion_descriptions[label])

    primary_emotion = top_emotions[0]['label'].lower() if top_emotions else "happy"
    emotion_label = map_emotion_to_music_auto(primary_emotion)
    st.markdown(f"🎼 Mapped emotion `{primary_emotion}` → **{emotion_label.capitalize()}** for music generation 🎵")

    if st.button("🎶 Generate Music"):
        with st.spinner("🎼 Generating high-quality music with MusicGen..."):
            musicgen.set_generation_params(duration=duration)
            output = musicgen.generate([f"{emotion_label} emotional background music"])
            output_path = f"outputs/{emotion_label}_musicgen.wav"
            audio_write(output_path[:-4], output[0].cpu(), sample_rate=32000)

        st.success("✅ Your music is ready!")
        st.audio(output_path)
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download WAV", f, file_name=os.path.basename(output_path))
