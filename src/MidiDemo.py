import argparse
import random
import torch

from MidiModel import EmotionLSTM, EmotionTransformer
from MidiUtils import create_vocab, sequence_to_midi, load_model
from MidiPreprocess import build_dataset
from MidiGenerate import generate, generate_using_transformer

SEQ_LENGTH = 32 # Use the same value as in training config
EMBED_DIM = 128
NHEAD = 4
DIM_FEEDFORWARD = 512
NUM_ENCODER_LAYERS = 3
NUM_EMOTIONS = 4
DROPOUT = 0.1
MAX_SEQ_LEN_LOAD = SEQ_LENGTH + 10 # Match training

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
seed_encoded = [note_to_int[n] for n in seed_notes[:SEQ_LENGTH]]

# Load model
vocab_size = len(note_to_int)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(EmotionLSTM, args.model_path, vocab_size, 64, 128, 16, 4).to("cuda")
lstm_model = load_model(
    EmotionLSTM,
    args.model_path,
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    hidden_dim=128,
    num_layers=4,
    num_emotions=NUM_EMOTIONS
).to(device)
transfromer_model = load_model(
    EmotionTransformer,
    args.model_path,
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

# Generate music - LSTM
emotion_id = emo_to_int[args.emotion]
result_sequence = generate(model, seed_encoded, emotion_id, length=args.length)

# Convert to MIDI - LSTM
sequence_to_midi(result_sequence, int_to_note, output_path=args.output)
print(f"Generated {args.output} for emotion: {args.emotion}")


# Generate music - Transformer
transformer_result_sequence = generate_using_transformer(
    transfromer_model,
    seed_encoded,
    emotion_id,
    seq_len=SEQ_LENGTH, # Pass seq_len
    length=args.length,
    temperature=0.8, # Example temperature for sampling
    top_k=50         # Example top-k sampling
)
# Convert to MIDI - Transformer
sequence_to_midi(transformer_result_sequence, int_to_note, output_path=args.output.replace('.mid', '_transformer.mid'))
print(f"Generated {args.output.replace('.mid', '_transformer.mid')} for emotion: {args.emotion}")
