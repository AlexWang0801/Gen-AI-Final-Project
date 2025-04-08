import streamlit as st
import random
import os
import torch
from midi2audio import FluidSynth

from src.MidiModel import EmotionLSTM
from src.MidiUtils import create_vocab, sequence_to_midi, load_model
from src.MidiPreprocess import build_dataset
from src.MidiGenerate import generate

import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


import sys
print("Python executable:", sys.executable)


# Emotion mapping
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


# UI setup
st.title("🎵 Emotion-Based Music Generator")
st.markdown("Generate music based on your selected emotion using a trained LSTM model and EMOPIA dataset.")

selected_display = st.selectbox("Choose an Emotion:", list(display_to_emo.keys()))
emotion = display_to_emo[selected_display]

if st.button("🎶 Generate Music"):
    with st.spinner("Generating your music..."):
        data, note_sequences, note_to_int, int_to_note, seed = load_resources()
        seed_encoded = [note_to_int[n] for n in seed[:32]]
        model = load_trained_model(len(note_to_int))

        result_sequence = generate(model, seed_encoded, emo_to_int[emotion], length=50)
        midi_output_path = f"outputs/{emotion}_streamlit.mid"
        sequence_to_midi(result_sequence, int_to_note, midi_output_path)

        # Convert MIDI to audio
        wav_output_path = f"outputs/{emotion}_streamlit.wav"
        fs = FluidSynth(sound_font="src/outputs/FluidR3_GM.sf2")
        fs.midi_to_audio(midi_output_path, wav_output_path)

    st.success("Done!")
    with open(midi_output_path, "rb") as f:
        st.download_button("⬇️ Download MIDI", f, file_name=os.path.basename(midi_output_path))
    st.audio(wav_output_path)
