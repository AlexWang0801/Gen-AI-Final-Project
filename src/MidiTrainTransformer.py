import torch
from torch.utils.data import DataLoader, TensorDataset
from MidiModel import EmotionTransformer
from MidiPreprocess import build_dataset
from MidiUtils import create_vocab, notes_to_input_target, save_model

# Paths
midi_dir = "../data/EMOPIA_2.1/midis/"
label_csv = "../data/EMOPIA_2.1/label.csv"

# Configuration
SEQ_LEN = 32
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 20
EMBED_DIM = 128
NHEAD = 4
DIM_FEEDFORWARD = 512
NUM_ENCODER_LAYERS = 3
EMOTION_DIM = 16
NUM_EMOTIONS = 4
DROPOUT = 0.1

# Load data
print("Loading and preprocessing data...")
data = build_dataset(midi_dir, label_csv)
data = [(notes, emo) for notes, emo in data if len(notes) > SEQ_LEN]

if not data:
     raise ValueError(f"No sequences found with length greater than {SEQ_LEN}. Check dataset or SEQ_LEN.")

note_sequences = [notes for notes, _ in data]
emotions = [emo for _, emo in data]
note_to_int, int_to_note = create_vocab(note_sequences)
emo_to_int = {"happy": 0, "tense": 1, "sad": 2, "peaceful": 3}
vocab_size = len(note_to_int)
print(f"Vocabulary size: {vocab_size}")

# Convert to tensors
print("Preparing tensors...")
X, y, e = [], [], []
for notes, emotion in zip(note_sequences, emotions):
    try:
        note_indices = [note_to_int[n] for n in notes]
        input_seqs, target_seqs = notes_to_input_target(note_indices, seq_len=SEQ_LEN)
        if input_seqs:
            X.extend(input_seqs)
            y.extend(target_seqs)
            e.extend([emo_to_int[emotion]] * len(input_seqs))
    except KeyError as ke:
        print(f"Warning: Key error during conversion (likely missing note '{ke}' in vocab). Skipping sequence.")
    except Exception as ex:
        print(f"Warning: Error processing sequence: {ex}. Skipping.")

if not X:
    raise ValueError("Failed to create any training samples. Check data and SEQ_LEN.")

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
e = torch.tensor(e, dtype=torch.long) # Emotion indices

# Check shapes
print(f"Input sequences shape (X): {X.shape}")
print(f"Emotion labels shape (e): {e.shape}")
print(f"Target notes shape (y): {y.shape}")

loader = DataLoader(TensorDataset(X, e, y), batch_size=BATCH_SIZE, shuffle=True)

# Train
print("Setting up model and training...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the Transformer model
model = EmotionTransformer(
    vocab_size=vocab_size,
    embed_dim=EMBED_DIM,
    nhead=NHEAD,
    dim_feedforward=DIM_FEEDFORWARD,
    num_encoder_layers=NUM_ENCODER_LAYERS,
    emotion_dim=EMBED_DIM, # Using Option 1: emotion_dim matches embed_dim
    num_emotions=NUM_EMOTIONS,
    dropout=DROPOUT,
    max_seq_len=SEQ_LEN + 10 # Max sequence length model can handle (can be larger than SEQ_LEN)
).to(device)

opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batch_count = 0
    for xb, eb, yb in loader:
        xb, eb, yb = xb.to(device), eb.to(device), yb.to(device)
        out = model(xb, eb)
        loss = loss_fn(out, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    print(f"Epoch {epoch+1}/{EPOCHS} Loss: {avg_loss:.4f}")

# Save the model
save_path = "outputs/emotion_transformer.pth"
save_model(model, path=save_path)
print(f"Model saved to {save_path}")
