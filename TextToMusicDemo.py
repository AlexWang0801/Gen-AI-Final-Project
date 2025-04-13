import streamlit as st
import os
import random
import json
import torch
from transformers import pipeline, BertTokenizerFast, BertForSequenceClassification
from midi2audio import FluidSynth
from src.MidiModel import EmotionLSTM
from src.MidiUtils import create_vocab, sequence_to_midi, load_model
from src.MidiPreprocess import build_dataset
from src.MidiGenerate import generate
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer('all-mpnet-base-v2')

music_labels = ["happy", "tense", "sad", "peaceful"]
music_embs = embedder.encode(music_labels, convert_to_tensor=True)

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

def map_emotion_to_music_auto(predicted_label: str) -> str:
    label = predicted_label.lower()
    
    if label in manual_map:
        return manual_map[label]
    
    emo_emb = embedder.encode(label, convert_to_tensor=True)
    cosine_scores = util.cos_sim(emo_emb, music_embs)[0]
    best_idx = int(torch.argmax(cosine_scores))
    return music_labels[best_idx]



# --- Page config must come first ---
st.set_page_config(page_title="🎵 Emotion Music Generator", layout="centered")

# --- Load emotion labels ---
with open("data/emotions.txt", "r") as f:
    emotions = [line.strip() for line in f]
emo_to_int = {"happy": 0, "tense": 1, "sad": 2, "peaceful": 3}  # for music mapping

# --- Load per-class optimal thresholds ---
with open("src/optimal_thresholds.json") as f:
    thresholds = json.load(f)

# --- Descriptions for UI ---
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

# --- Load music generation model & data ---
midi_dir = "./data/EMOPIA_2.1/midis/"
label_csv = "./data/EMOPIA_2.1/label.csv"
model_path = "./src/outputs/emotion_lstm.pth"

@st.cache_data
def load_music_resources():
    data = build_dataset(midi_dir, label_csv)
    note_sequences = [notes for notes, _ in data]
    note_to_int, int_to_note = create_vocab(note_sequences)
    seed = random.choice(note_sequences)
    return data, note_sequences, note_to_int, int_to_note, seed

def generate(model, seed, emotion_id, length=100, temperature=1.0):
    model.eval()
    generated = seed.copy()
    input_seq = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(next(model.parameters()).device)
    emotion_tensor = torch.tensor([emotion_id]).to(input_seq.device)

    for _ in range(length):
        output = model(input_seq, emotion_tensor)
        logits = model(input_seq, emotion_tensor)
        probs = torch.softmax(logits / temperature, dim=-1)
        next_note = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_note)

        input_seq = torch.tensor(generated[-len(seed):], dtype=torch.long).unsqueeze(0).to(input_seq.device)

    return generated

@st.cache_resource
def load_music_model(vocab_size):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return load_model(EmotionLSTM, model_path, vocab_size, 64, 128, 16, 4).to(device)

# --- UI ---
st.title("🎵 Emotion-to-Music Generator")
st.markdown("""
Enter a sentence describing how you feel, and this app will:
1. Detect your emotion using a fine-tuned BERT model 🤖  
2. Generate a piece of music that matches your mood 🎶  
3. Let you listen to and download your emotional melody 🎧  
""")

user_input = st.text_area("💬 What are you feeling?",
    placeholder="e.g. I feel calm and peaceful watching the rain fall...")
temperature = st.slider(
    "🎛️ Creativity Level (Higher = More Variation)", 
    min_value=0.7, 
    max_value=1.5, 
    value=1.0, 
    step=0.1
)

if user_input:
    with st.spinner("🔍 Detecting emotion..."):
        raw_scores = emotion_classifier(user_input)[0]

        # Apply per-class threshold filtering
        predicted_emotions = []
        for item in raw_scores:
            label = item["label"]
            score = item["score"]
            if score > thresholds.get(label, 0.5):
                predicted_emotions.append({"label": label, "score": score})

        # Sort top
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
        with st.spinner("🎼 Composing your emotional melody..."):
            data, note_sequences, note_to_int, int_to_note, seed = load_music_resources()
            random_start = random.randint(0, max(0, len(seed) - 32))
            seed_encoded = [note_to_int[n] for n in seed[random_start:random_start + 32]]
            model = load_music_model(len(note_to_int))
            result_sequence = generate(model, seed_encoded, emo_to_int[emotion_label], length=100, temperature=temperature)

            midi_output_path = f"outputs/{emotion_label}_streamlit.mid"
            wav_output_path = f"outputs/{emotion_label}_streamlit.wav"

            sequence_to_midi(result_sequence, int_to_note, midi_output_path)
            fs = FluidSynth(sound_font="src/outputs/FluidR3_GM.sf2")
            fs.midi_to_audio(midi_output_path, wav_output_path)

        st.success("✅ Your music is ready!")
        st.audio(wav_output_path)
        with open(midi_output_path, "rb") as f:
            st.download_button("⬇️ Download MIDI", f, file_name=os.path.basename(midi_output_path))
