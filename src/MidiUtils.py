def create_vocab(note_sequences):
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
    from music21 import stream, note, chord
    midi_stream = stream.Stream()
    for idx in note_sequence:
        n = int_to_note[idx]
        if "." in n:
            chord_notes = [int(x) for x in n.split(".")]
            midi_stream.append(chord.Chord(chord_notes))
        else:
            midi_stream.append(note.Note(n))
    midi_stream.write("midi", fp=output_path)


def save_model(model, path="model.pth"):
    import torch
    torch.save(model.state_dict(), path)


def load_model(model_class, path, *args, **kwargs):
    import torch
    model = model_class(*args, **kwargs)
    model.load_state_dict(torch.load(path))
    return model
