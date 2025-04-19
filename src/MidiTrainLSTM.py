import torch
from torch.utils.data import DataLoader, TensorDataset
from MidiModel import EmotionLSTM
from MidiPreprocess import build_dataset
from MidiUtils import create_vocab, notes_to_input_target, save_model

# Paths
midi_dir = "../data/EMOPIA_2.1/midis/"
label_csv = "../data/EMOPIA_2.1/label.csv"

# Load data
data = build_dataset(midi_dir, label_csv)
note_sequences = [notes for notes, _ in data]
emotions = [emo for _, emo in data]
note_to_int, int_to_note = create_vocab(note_sequences)
emo_to_int = {"happy": 0, "tense": 1, "sad": 2, "peaceful": 3}

# Convert to tensors
X, y, e = [], [], []
for notes, emotion in zip(note_sequences, emotions):
    note_indices = [note_to_int[n] for n in notes]
    input_seq, target_seq = notes_to_input_target(note_indices)
    X.extend(input_seq)
    y.extend(target_seq)
    e.extend([emo_to_int[emotion]] * len(input_seq))

X = torch.tensor(X, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)
e = torch.tensor(e, dtype=torch.long)
loader = DataLoader(TensorDataset(X, e, y), batch_size=64, shuffle=True)

# Train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionLSTM(len(note_to_int), 64, 128, 16, 4).to(device)

for xb, eb, yb in loader:
    xb, eb, yb = xb.to(device), eb.to(device), yb.to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    total = 0
    for xb, eb, yb in loader:
        # Move tensors to the appropriate device
        xb, eb, yb = xb.to(device), eb.to(device), yb.to(device)
        out = model(xb, eb)
        loss = loss_fn(out, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch+1} Loss: {total:.4f}")

# Save the model
save_model(model, path="outputs/emotion_lstm.pth")
