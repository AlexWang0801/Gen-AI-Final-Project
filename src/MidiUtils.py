import torch
import os
from music21 import stream, note, chord


def create_vocab(note_sequences):
    if not note_sequences:
        raise ValueError("Note sequences are empty. Cannot create vocabulary.")
    # Flatten the list of note sequences and create a unique mapping
    all_notes = [n for seq in note_sequences for n in seq]
    unique = sorted(set(all_notes))
    note_to_int = {n: i for i, n in enumerate(unique)}
    int_to_note = {i: n for n, i in note_to_int.items()}
    return note_to_int, int_to_note


def notes_to_input_target(note_sequence, seq_len=32):
    inputs, targets = [], []
    for i in range(len(note_sequence) - seq_len):
        inputs.append(note_sequence[i:i+seq_len])
        targets.append(note_sequence[i+seq_len])
    return inputs, targets


def sequence_to_midi(note_sequence, int_to_note, output_path="generated.mid"):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    midi_stream = stream.Stream()
    for idx in note_sequence:
        n = int_to_note[idx]
        try:
            if "." in n:
                chord_notes = [int(x) for x in n.split(".")]
                midi_stream.append(chord.Chord(chord_notes))
            else:
                midi_stream.append(note.Note(n))
        except Exception as e:
            print(f"Error converting note/chord {n}: {e}")
    midi_stream.write("midi", fp=output_path)


def save_model(model, path="model.pth"):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model_class, path, *args, **kwargs):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file {path} not found.")
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model
