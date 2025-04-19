import torch
import os
import random
import time

# Import your project modules
# Ensure these paths are correct relative to where you run this script
from MidiModel import EmotionLSTM, EmotionTransformer
from MidiUtils import create_vocab, sequence_to_midi, load_model
from MidiPreprocess import build_dataset
from MidiGenerate import generate_using_transformer
# --- Configuration ---

# Model Hyperparameters & Paths (MUST match training config)
# Reusing the config structure from the Streamlit app
SEQ_LEN = 32

LSTM_CONFIG = {
    "model_path": "outputs/emotion_lstm.pth",
    "args": [64, 128, 16, 4], # vocab_size, embed_dim, hidden_dim, emotion_dim, num_emotions
    "kwargs": {}
}

TRANSFORMER_CONFIG = {
    "model_path": "outputs/emotion_transformer.pth",
    "args": [], # vocab_size added later
    "kwargs": {
        "embed_dim": 128, "nhead": 4, "dim_feedforward": 512,
        "num_encoder_layers": 3, "emotion_dim": 128, "num_emotions": 4,
        "dropout": 0.1, "max_seq_len": SEQ_LEN + 10
    }
}

MODEL_CONFIGS = {
    "EmotionLSTM": LSTM_CONFIG, # Use class name as key for clarity
    "EmotionTransformer": TRANSFORMER_CONFIG
}

# Data Paths
DATA_DIR = "../data"
MIDI_SUBDIR = "EMOPIA_2.1/midis"
LABEL_FILENAME = "EMOPIA_2.1/label.csv"
midi_dir = os.path.join(DATA_DIR, MIDI_SUBDIR)
label_csv = os.path.join(DATA_DIR, LABEL_FILENAME)

# Generation Parameters
NUM_FILES_PER_CONDITION = 5
GENERATION_LENGTH = 150
TEMPERATURE = 0.8
# TOP_K = 50

# Output Directory
BASE_OUTPUT_DIR = './evaluation_output'

# Emotions to generate for
EMOTIONS_TO_GENERATE = ["happy", "sad", "tense", "peaceful"]
emo_to_int = {"happy": 0, "tense": 1, "sad": 2, "peaceful": 3}


def batch_generate():
    """
    Main function to generate MIDI files for all models and emotions.
    """
    print("--- Starting Batch MIDI Generation ---")

    print("Loading dataset and vocabulary...")
    try:
        data = build_dataset(midi_dir, label_csv)
        data = [(notes, emo) for notes, emo in data if len(notes) > SEQ_LEN]
        if not data:
            print(f"Error: No sequences found with length greater than {SEQ_LEN}. Cannot proceed.")
            return

        note_sequences = [notes for notes, _ in data]
        note_to_int, int_to_note = create_vocab(note_sequences)
        vocab_size = len(note_to_int)
        print(f"Vocabulary Size: {vocab_size}")
        # Keep original sequences for seeding
        seed_pool = note_sequences
        if not seed_pool:
             print("Error: Seed pool is empty after filtering. Cannot generate.")
             return
        print("Dataset and vocabulary loaded.")
    except FileNotFoundError:
        print(f"Error: Data files not found. Check paths:")
        print(f"  MIDI Dir: '{midi_dir}'")
        print(f"  Label CSV: '{label_csv}'")
        return
    except Exception as e:
        print(f"Error loading resources: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    total_files_generated = 0
    start_time = time.time()

    for model_name, config in MODEL_CONFIGS.items():
        print(f"\n--- Processing Model: {model_name} ---")
        model_class = EmotionLSTM if model_name == "EmotionLSTM" else EmotionTransformer
        model_path = config["model_path"]

        # Prepare args/kwargs for model loading
        args = [vocab_size] + config.get("args", [])
        kwargs = config.get("kwargs", {})

        # Load the model
        try:
            print(f"Loading model from: {model_path}")
            model = load_model(model_class, model_path, *args, **kwargs)
            model.to(device)
            model.eval() # Set model to evaluation mode
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {model_path}. Skipping this model.")
            continue
        except Exception as e:
            print(f"Error loading model {model_name}: {e}. Skipping this model.")
            continue

        # Loop through emotions
        for emotion in EMOTIONS_TO_GENERATE:
            print(f"  Generating for Emotion: {emotion}")
            emotion_id = emo_to_int[emotion]

            # Create output directory for this condition
            current_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name, emotion)
            os.makedirs(current_output_dir, exist_ok=True)
            print(f"    Output directory: {current_output_dir}")

            files_generated_for_condition = 0
            for i in range(NUM_FILES_PER_CONDITION):
                print(f"    Generating file {i+1}/{NUM_FILES_PER_CONDITION}...")
                try:
                    # Select and prepare seed
                    seed_sequence = random.choice(seed_pool)
                    # Ensure seed is long enough (should be guaranteed by earlier filter)
                    seed_encoded = [note_to_int[n] for n in seed_sequence[:SEQ_LEN]]

                    # Generate note indices
                    result_sequence = generate_using_transformer(
                        model=model,
                        input_seq_indices=seed_encoded,
                        emotion_label=emotion_id,
                        seq_len=SEQ_LEN,
                        length=GENERATION_LENGTH,
                        temperature=TEMPERATURE
                    )

                    output_filename = f"{model_name}_{emotion}_gen_{i+1}.mid"
                    output_path = os.path.join(current_output_dir, output_filename)
                    sequence_to_midi(result_sequence, int_to_note, output_path=output_path)
                    files_generated_for_condition += 1
                    total_files_generated += 1

                except KeyError as ke:
                    print(f"Error during generation (file {i+1}): Note '{ke}' from seed not in vocabulary. Skipping this file.")
                except Exception as e:
                    print(f"Error during generation or saving (file {i+1}): {e}. Skipping this file.")

            print(f"Generated {files_generated_for_condition} files for {emotion}.")

    end_time = time.time()
    duration = end_time - start_time
    print("\n--- Batch Generation Complete ---")
    print(f"Total files generated: {total_files_generated}")
    print(f"Total time taken: {duration:.2f} seconds")
    print(f"Generated files saved in base directory: {BASE_OUTPUT_DIR}")

if __name__ == "__main__":
    batch_generate()
