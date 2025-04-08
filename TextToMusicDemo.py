import streamlit as st
import random
import os
from transformers import pipeline

from src.MidiModel import EmotionLSTM
from src.MidiUtils import create_vocab, sequence_to_midi, load_model
from src.MidiPreprocess import build_dataset
from src.MidiGenerate import generate

# Emotion mapping
emotion_mapping = {
    "joy": "happy",
    "excitement": "happy",
    "contentment": "peaceful",
    "calm": "peaceful",
    "sadness": "sad",
    "grief": "sad",
    "nervousness": "tense",
    "fear": "tense",
    "anger": "tense",
    "disgust": "tense",
    "surprise": "tense"
}
emo_to_int = {"happy": 0, "tense": 1, "sad": 2, "peaceful": 3}

# Paths
midi_dir = "data/midis"
label_csv = r"data\EMOPIA_2.1\label.csv"
model_path = "outputs/emotion_lstm.pth"

# Load dataset & vocab only once
@st.cache_data
def load_resources():
    data = build_dataset(midi_dir, label_csv)
    note_sequences = [notes for notes, _ in data]
    note_to_int, int_to_note = create_vocab(note_sequences)
    seed = random.choice(note_sequences)
    return data, note_sequences, note_to_int, int_to_note, seed

# Load model once
@st.cache_resource
def load_trained_model(vocab_size):
    return load_model(EmotionLSTM, model_path, vocab_size, 64, 128, 16, 4).to("cuda")

# Emotion detection pipeline
emotion_classifier = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")

# UI setup
st.title("🎵 Emotion-Based Music Generator")
st.markdown("Generate music based on the emotion detected in your text using a trained LSTM model and EMOPIA dataset.")

# User input
user_input = st.text_area("Enter your thoughts or feelings:")

if st.button("🔍 Submit"):
    with st.spinner("Detecting emotion..."):
        results = emotion_classifier(user_input)
        top_emotion = results[0]['label'].lower()

        # Map to 4-class emotion
        mapped_emotion = emotion_mapping.get(top_emotion, "happy")  # default to happy
        st.write(f"Detected Emotion: **{top_emotion.capitalize()}** → **{mapped_emotion.capitalize()}**")

    if st.button("🎶 Generate Music"):
        with st.spinner("Generating your music..."):
            data, note_sequences, note_to_int, int_to_note, seed = load_resources()
            seed_encoded = [note_to_int[n] for n in seed[:32]]
            model = load_trained_model(len(note_to_int))

            result_sequence = generate(model, seed_encoded, emo_to_int[mapped_emotion], length=50)
            output_path = f"outputs/{mapped_emotion}_streamlit.mid"
            sequence_to_midi(result_sequence, int_to_note, output_path)

        st.success("Done!")
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download MIDI", f, file_name=os.path.basename(output_path))
        st.audio(output_path)
