import argparse
import random

from MidiModel import EmotionLSTM, EmotionTransformer
from MidiUtils import create_vocab, sequence_to_midi, load_model
from MidiPreprocess import build_dataset
from MidiGenerate import generate

# Emotion label map
emo_to_int = {"happy": 0, "tense": 1, "sad": 2, "peaceful": 3}

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--emotion', type=str, required=True, choices=emo_to_int.keys())
parser.add_argument('--model_path', type=str, default='outputs/emotion_lstm.pth')
parser.add_argument('--output', type=str, default='outputs/generated.mid')
parser.add_argument('--length', type=int, default=50)
args = parser.parse_args()

# Load data and vocab
midi_dir = "data/midis"
label_csv = "data/label.csv"
data = build_dataset(midi_dir, label_csv)
note_sequences = [notes for notes, _ in data]
note_to_int, int_to_note = create_vocab(note_sequences)

# Create seed sequence
seed_notes = random.choice(note_sequences)
seed_encoded = [note_to_int[n] for n in seed_notes[:32]]

# Load model
vocab_size = len(note_to_int)
model = load_model(EmotionLSTM, args.model_path, vocab_size, 64, 128, 16, 4).to("cuda")

# Generate music
emotion_id = emo_to_int[args.emotion]
result_sequence = generate(model, seed_encoded, emotion_id, length=args.length)

# Convert to MIDI
sequence_to_midi(result_sequence, int_to_note, output_path=args.output)
print(f"Generated {args.output} for emotion: {args.emotion}")


# ... imports and arg parsing ...
from MidiModel import EmotionTransformer # Import Transformer

# ... Load data and vocab ...
SEQ_LENGTH = 32 # Use the same value as in training config
seed_encoded = [note_to_int[n] for n in seed_notes[:SEQ_LENGTH]]
vocab_size = len(note_to_int)

# --- Load model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = args.model_path # e.g., 'outputs/emotion_transformer.pth'

# !!! IMPORTANT: Pass ALL hyperparameters used during training !!!
# (These should ideally come from a saved config)
EMBED_DIM = 128
NHEAD = 4
DIM_FEEDFORWARD = 512
NUM_ENCODER_LAYERS = 3
NUM_EMOTIONS = 4
DROPOUT = 0.1
MAX_SEQ_LEN_LOAD = SEQ_LENGTH + 10 # Match training

try:
    model = load_model(
        EmotionTransformer, # Load the correct class
        model_path,
        # Pass hyperparameters needed by __init__
        vocab_size=vocab_size,
        embed_dim=EMBED_DIM,
        nhead=NHEAD,
        dim_feedforward=DIM_FEEDFORWARD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        emotion_dim=EMBED_DIM, # Matched embed_dim
        num_emotions=NUM_EMOTIONS,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN_LOAD
    ).to(device)
    print(f"Loaded model from {model_path}")

except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)


# --- Generate music ---
emotion_id = emo_to_int[args.emotion]
print(f"Generating music for emotion: {args.emotion} (ID: {emotion_id})")

# Pass seq_len to generate function
result_sequence = generate(
    model,
    seed_encoded,
    emotion_id,
    seq_len=SEQ_LENGTH, # Pass seq_len
    length=args.length,
    temperature=0.8, # Example temperature for sampling
    top_k=50         # Example top-k sampling
)

# --- Convert to MIDI ---
# ... (rest of the script) ...