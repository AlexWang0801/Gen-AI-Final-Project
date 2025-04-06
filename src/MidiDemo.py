import argparse
import random

from src.MidiModel import EmotionLSTM
from src.MidiUtils import create_vocab, sequence_to_midi, load_model
from src.MidiPreprocess import build_dataset
from src.MidiGenerate import generate

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
