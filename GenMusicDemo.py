import streamlit as st
import random
import os
import torch
import pandas as pd

from src.MidiModel import EmotionLSTM, EmotionTransformer
from src.MidiUtils import create_vocab, sequence_to_midi, load_model
from src.MidiPreprocess import build_dataset
from src.MidiGenerate import generate, generate_using_transformer

# Model Hyperparameters (MUST match training configuration for each model)
SEQ_LEN = 32 # Define sequence length consistently

LSTM_CONFIG = {
    "model_path": "src/outputs/emotion_lstm.pth",
    "embed_dim": 64,
    "hidden_dim": 128,
    "emotion_dim": 16,
    "num_emotions": 4,
    "args": [64, 128, 16, 4],
    "kwargs": {}
}

TRANSFORMER_CONFIG = {
    "model_path": "src/outputs/emotion_transformer.pth",
    "embed_dim": 128,
    "nhead": 4,             
    "dim_feedforward": 512, 
    "num_encoder_layers": 3,
    "emotion_dim": 128, 
    "num_emotions": 4,
    "dropout": 0.1,
    "max_seq_len": SEQ_LEN + 10, 
    "args": [], 
    "kwargs": { 
        "embed_dim": 128,
        "nhead": 4,
        "dim_feedforward": 512,
        "num_encoder_layers": 3,
        "emotion_dim": 128,
        "num_emotions": 4,
        "dropout": 0.1,
        "max_seq_len": SEQ_LEN + 10
    }
}

MODEL_CONFIGS = {
    "LSTM": LSTM_CONFIG,
    "Transformer": TRANSFORMER_CONFIG
}

# Data Paths (Using os.path.join for better portability)
DATA_DIR = "data" # Base directory for data
MIDI_SUBDIR = "EMOPIA_2.1/midis" # Subdirectory for MIDIs
LABEL_FILENAME = "EMOPIA_2.1/label.csv" # Label filename

# Construct full paths
midi_dir = os.path.join(DATA_DIR, MIDI_SUBDIR)
label_csv = os.path.join(DATA_DIR, LABEL_FILENAME)

# Output directory for generated files
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# Emotion mapping
display_to_emo = {"😊 Happy": "happy", "😨 Tense": "tense", "😢 Sad": "sad", "😌 Peaceful": "peaceful"}
emo_to_int = {"happy": 0, "tense": 1, "sad": 2, "peaceful": 3}

# --- Helper Functions ---

# Load dataset & vocab only once
@st.cache_data
def load_resources(midi_path, label_path):
    """Loads dataset, creates vocabulary, and selects a random seed sequence."""
    print("Loading resources (dataset, vocab)...") # Add print statement for debugging cache
    try:
        data = build_dataset(midi_path, label_path)
        # Filter sequences shorter than SEQ_LEN + 1
        data = [(notes, emo) for notes, emo in data if len(notes) > SEQ_LEN]
        if not data:
            st.error(f"No sequences found with length greater than {SEQ_LEN}. Check dataset or SEQ_LEN setting.")
            return None, None, None, None, None

        note_sequences = [notes for notes, _ in data]
        note_to_int, int_to_note = create_vocab(note_sequences)
        # Select a random seed sequence *after* filtering
        seed_sequence = random.choice(note_sequences)
        print("Resources loaded successfully.")
        return data, note_sequences, note_to_int, int_to_note, seed_sequence
    except FileNotFoundError:
        st.error(f"Error: Data files not found. Expected MIDI dir: '{midi_path}' and Label CSV: '{label_path}'")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading resources: {e}")
        return None, None, None, None, None

# Load model once per selected model type
# The key here is that the function arguments determine the cache key
@st.cache_resource
def load_trained_model(model_type, vocab_size):
    """Loads the specified trained model (LSTM or Transformer)."""
    print(f"Loading model: {model_type} (Vocab size: {vocab_size})") # Debug print
    if model_type not in MODEL_CONFIGS:
        st.error(f"Unknown model type: {model_type}")
        return None

    config = MODEL_CONFIGS[model_type]
    model_path = config["model_path"]
    model_class = EmotionLSTM if model_type == "LSTM" else EmotionTransformer

    # Prepare args and kwargs, adding vocab_size
    args = [vocab_size] + config.get("args", [])
    kwargs = config.get("kwargs", {})

    # Dynamic device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Using device: {device}") # Inform user

    try:
        # Use the utility function to load the model
        # Pass the class, path, and *unpacked* args/kwargs
        model = load_model(model_class, model_path, *args, **kwargs)
        model.to(device) # Move model to the selected device
        print(f"Model {model_type} loaded successfully from {model_path}.")
        return model, device # Return model and device
    except FileNotFoundError:
        st.error(f"Error: Model file not found at {model_path}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        # Add more detailed error info if possible
        st.exception(e)
        return None, None

# --- Streamlit UI Setup ---

st.set_page_config(layout="wide") # Use wider layout
st.title("🎵 Emotion-Based Music Generator")
st.markdown("Generate MIDI music based on your selected emotion using different deep learning models trained on the EMOPIA dataset.")

# --- Sidebar for Options ---
st.sidebar.header("Generation Options")

# Model Selection
selected_model_type = st.sidebar.selectbox(
    "Choose Model Architecture:",
    list(MODEL_CONFIGS.keys()) # Get model names from config keys
)

# Emotion Selection
selected_display_emotion = st.sidebar.selectbox(
    "Choose an Emotion:",
    list(display_to_emo.keys())
)
emotion = display_to_emo[selected_display_emotion]

# Generation Length
generation_length = st.sidebar.slider("Generation Length (notes):", 50, 500, 100)

# Sampling Temperature
temperature = st.sidebar.slider("Sampling Temperature (randomness):", 0.1, 2.0, 0.8, 0.05)


# --- Main Area ---

# Load resources (cached)
resources = load_resources(midi_dir, label_csv)

if resources and all(r is not None for r in resources):
    data, note_sequences, note_to_int, int_to_note, seed_sequence = resources
    vocab_size = len(note_to_int)

    st.write(f"Dataset loaded. Vocabulary size: {vocab_size}. Using random seed sequence.")

    # Generate Button
    if st.sidebar.button(f"🎶 Generate Music ({selected_model_type})"):
        st.subheader(f"Generating {selected_display_emotion} Music using {selected_model_type}...")

        # Load the selected model (cached)
        model, device = load_trained_model(selected_model_type, vocab_size)

        if model is not None and device is not None:
            with st.spinner("Generating your music... Please wait."):
                # Prepare seed sequence
                try:
                    # Ensure seed sequence is long enough
                    if len(seed_sequence) < SEQ_LEN:
                         st.warning(f"Selected random seed sequence is shorter than SEQ_LEN ({SEQ_LEN}). Using the full seed.")
                         seed_encoded = [note_to_int[n] for n in seed_sequence]
                    else:
                         seed_encoded = [note_to_int[n] for n in seed_sequence[:SEQ_LEN]]

                    # Generate sequence
                    # Ensure 'generate' function signature matches: model, seed, emotion_id, seq_len, length, temperature etc.
                    result_sequence = generate_using_transformer(
                        model=model,
                        input_seq_indices=seed_encoded,
                        emotion_label=emo_to_int[emotion],
                        seq_len=SEQ_LEN, # Pass sequence length
                        length=generation_length,
                        temperature=temperature
                    )

                    # Convert to MIDI
                    output_filename = f"{selected_model_type}_{emotion}_{random.randint(1000,9999)}.mid"
                    output_path = os.path.join(OUTPUT_DIR, output_filename)
                    sequence_to_midi(result_sequence, int_to_note, output_path=output_path)

                    st.success("🎉 Music Generation Complete!")
                    st.markdown(f"**Generated File:** `{output_filename}`")

                    # Provide download and playback
                    try:
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="⬇️ Download MIDI File",
                                data=f,
                                file_name=output_filename,
                                mime="audio/midi"
                            )
                        st.audio(output_path)
                    except FileNotFoundError:
                        st.error(f"Could not find the generated MIDI file at {output_path} for playback/download.")
                    except Exception as e:
                        st.error(f"Error providing file download/playback: {e}")

                except KeyError as ke:
                    st.error(f"Error during generation: Note '{ke}' from the seed sequence not found in the vocabulary. Please clear cache and retry.")
                except Exception as e:
                    st.error(f"An error occurred during music generation: {e}")
                    st.exception(e) # Show detailed traceback in app
        else:
            st.error("Model could not be loaded. Please check configuration and file paths.")
else:
    st.error("Failed to load necessary data resources. Please check the configured data paths and file integrity.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed using PyTorch & Streamlit.")

