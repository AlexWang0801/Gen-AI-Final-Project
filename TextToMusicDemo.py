import streamlit as st
import random
import os
from transformers import pipeline
import torch
from midi2audio import FluidSynth

from src.MidiModel import EmotionLSTM
from src.MidiUtils import create_vocab, sequence_to_midi, load_model
from src.MidiPreprocess import build_dataset
from src.MidiGenerate import generate

# Mapping emotions for music generation (you can adjust this as needed)
display_to_emo = {"😊 Happy": "happy", "😨 Tense": "tense", "😢 Sad": "sad", "😌 Peaceful": "peaceful"}
emo_to_int = {"happy": 0, "tense": 1, "sad": 2, "peaceful": 3}

# Paths
midi_dir = "./data/EMOPIA_2.1/midis/"
label_csv = "./data/EMOPIA_2.1/label.csv"
model_path = "./src/outputs/emotion_lstm.pth"

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_model(
        EmotionLSTM, model_path, vocab_size, 64, 128, 16, 4
    ).to(device)


# Initialize emotion detection pipeline from Hugging Face
emotion_classifier = pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')

# UI setup
st.title("🎵 Emotion-Based Music Generator")
st.markdown("Generate music based on the detected emotion in your text using a trained LSTM model and EMOPIA dataset.")

# Text input for user to enter their text
user_input = st.text_area("Enter some text to detect emotion:")

if user_input:
    # Detect emotion from the input text using the transformers pipeline
    emotion_results = emotion_classifier(user_input)
    detected_emotion = emotion_results[0]['label'].lower()  # Extract emotion from the result

    st.write(f"Detected Emotion: {detected_emotion.capitalize()}")

    # Map the detected emotion to the corresponding music generation
    if detected_emotion in emo_to_int:
        emotion_label = detected_emotion
    else:
        emotion_label = "happy"  # Default to 'happy' if the emotion isn't recognized

    # Show emotion selection dropdown
    selected_display = st.selectbox("Choose an Emotion:", list(emo_to_int.keys()), index=list(emo_to_int.keys()).index(emotion_label))

    # Generate music based on the detected emotion
    if st.button("🎶 Generate Music"):
        with st.spinner("Generating your music..."):
            data, note_sequences, note_to_int, int_to_note, seed = load_resources()
            seed_encoded = [note_to_int[n] for n in seed[:32]]
            model = load_trained_model(len(note_to_int))

            result_sequence = generate(model, seed_encoded, emo_to_int[emotion_label], length=50)
            midi_output_path = f"outputs/{emotion_label}_streamlit.mid"
            sequence_to_midi(result_sequence, int_to_note, midi_output_path)

            # Convert MIDI to audio
            wav_output_path = f"outputs/{emotion_label}_streamlit.wav"
            fs = FluidSynth(sound_font="src/outputs/FluidR3_GM.sf2")
            fs.midi_to_audio(midi_output_path, wav_output_path)

        st.success("Done!")
        with open(midi_output_path, "rb") as f:
            st.download_button("⬇️ Download MIDI", f, file_name=os.path.basename(midi_output_path))
        st.audio(wav_output_path)
