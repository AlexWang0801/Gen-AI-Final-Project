import os
import pandas as pd
from music21 import converter, instrument, note, chord


def extract_notes(file_path):
    midi = converter.parse(file_path)
    notes = []
    try:
        parts = instrument.partitionByInstrument(midi)
        for part in parts.parts:
            for el in part.recurse():
                if isinstance(el, note.Note):
                    notes.append(str(el.pitch))
                elif isinstance(el, chord.Chord):
                    notes.append('.'.join(str(n) for n in el.normalOrder))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return notes


def build_dataset(midi_dir, label_csv):
    label_df = pd.read_csv(label_csv)
    label_map = {1: "happy", 2: "tense", 3: "sad", 4: "peaceful"}
    label_df["emotion"] = label_df["4Q"].map(label_map)

    data = []
    for _, row in label_df.iterrows():
        midi_file = os.path.join(midi_dir, f"{row['ID']}.mid")
        if os.path.exists(midi_file):
            notes = extract_notes(midi_file)
            if len(notes) > 32:
                data.append((notes, row["emotion"]))

    return data
