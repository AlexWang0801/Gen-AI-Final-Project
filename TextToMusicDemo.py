import streamlit as st
import os
import json
import torch
from transformers import pipeline
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

# --- Page config ---
st.set_page_config(page_title="🎵 Emotion Music Generator (MusicGen)", layout="centered")

# --- Load emotion labels ---
with open("data/emotions.txt", "r") as f:
    emotions = [line.strip() for line in f]

# --- Load thresholds ---
with open("src/optimal_thresholds.json") as f:
    thresholds = json.load(f)

# --- Manual mapping ---
manual_map = {
    "anger": "tense", "annoyance": "tense", "fear": "tense", "grief": "sad",
    "sadness": "sad", "joy": "happy", "gratitude": "happy", "relief": "peaceful"
}

emotion_descriptions = {
    "joy": "You're feeling joyful and cheerful! 😊", "sadness": "Sounds like you're feeling a bit down. 😢",
    "anger": "You're expressing anger or frustration. 😠", "gratitude": "You're feeling thankful and appreciative. 🙏",
    "love": "Your message shows love and affection. ❤️", "fear": "There’s a sense of fear or anxiety. 😨",
    "surprise": "That caught you off guard! 😲", "neutral": "A calm, neutral mood. 😐",
    "peaceful": "You're calm and at ease. 🌿", "tense": "There’s tension or unease. 😬",
}

@st.cache_resource
def load_emotion_model():
    model = "bhadresh-savani/distilbert-base-uncased-emotion"
    return pipeline("text-classification", model=model, tokenizer=model, top_k=None, function_to_apply="sigmoid")

@st.cache_resource
def load_musicgen_model():
    return MusicGen.get_pretrained('facebook/musicgen-small')

emotion_classifier = load_emotion_model()
musicgen = load_musicgen_model()

# --- Embedding for emotion mapping ---
from sentence_transformers import SentenceTransformer, util
embedder = SentenceTransformer('all-mpnet-base-v2')
music_labels = ["happy", "tense", "sad", "peaceful"]
music_embs = embedder.encode(music_labels, convert_to_tensor=True)

def map_emotion_to_music_auto(predicted_label: str) -> str:
    label = predicted_label.lower()
    if label in manual_map:
        return manual_map[label]
    emo_emb = embedder.encode(label, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emo_emb, music_embs)[0]
    return music_labels[int(torch.argmax(cosine_scores))]

# --- Session state ---
for key in ["generated", "output_path", "emotion_label", "user_input"]:
    if key not in st.session_state:
        st.session_state[key] = ""

# --- UI ---
st.title("🎵 Emotion-to-Music Generator (with MusicGen)")
st.markdown("""
Enter a sentence describing how you feel, and this app will:
1. Detect your emotion using a fine-tuned BERT model 🤖  
2. Generate a piece of **audio** music with Meta's MusicGen 🎶  
3. Let you listen to and download your emotional melody 🎧  
""")

user_input = st.text_area("💬 What are you feeling?", placeholder="e.g. I feel calm and peaceful watching the rain fall...")
duration = 10

# --- Main logic ---
if user_input:
    with st.spinner("🔍 Detecting emotion..."):
        raw_scores = emotion_classifier(user_input)[0]
        predicted_emotions = [{"label": i["label"], "score": i["score"]} for i in raw_scores if i["score"] > thresholds.get(i["label"], 0.5)]
        top_emotions = sorted(predicted_emotions, key=lambda x: x["score"], reverse=True)[:2]

    st.subheader("🎯 Detected Emotions:")
    for emo in top_emotions:
        st.markdown(f"**• {emo['label'].capitalize()} ({emo['score']:.2f})**")
        st.caption(emotion_descriptions.get(emo['label'], ""))

    primary_emotion = top_emotions[0]["label"].lower() if top_emotions else "happy"
    emotion_label = map_emotion_to_music_auto(primary_emotion)
    st.markdown(f"🎼 Mapped emotion `{primary_emotion}` → **{emotion_label.capitalize()}** for music generation 🎵")

    if st.button("🎶 Generate Music"):
        with st.spinner("🎼 Generating high-quality music with MusicGen..."):
            musicgen.set_generation_params(duration=duration)
            output = musicgen.generate([f"{emotion_label} emotional background music"])
            output_path = f"outputs/{emotion_label}_musicgen.wav"
            audio_write(output_path[:-4], output[0].cpu(), sample_rate=32000)

        st.session_state["generated"] = True
        st.session_state["output_path"] = output_path
        st.session_state["emotion_label"] = emotion_label
        st.session_state["user_input"] = user_input
        st.rerun()

# --- Result display ---
if st.session_state["generated"]:
    st.success("✅ Your music is ready!")
    st.audio(st.session_state["output_path"])
    with open(st.session_state["output_path"], "rb") as f:
        st.download_button("⬇️ Download WAV", f, file_name=os.path.basename(st.session_state["output_path"]))
