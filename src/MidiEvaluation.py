import music21
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# --- Configuration ---
# Adjust this path to where your generated MIDI files are stored
# Assume a structure like: base_dir / model_name / emotion_name / file.mid
MIDI_BASE_DIR = './generated_midi_files' # <--- CHANGE THIS PATH

# List of models and emotions you generated files for
MODELS = ['EmotionLSTM', 'EmotionTransformer']
EMOTIONS = ['happy', 'sad', 'tense', 'peaceful']


def extract_midi_features(midi_path):
    """
    Extracts basic pitch features from a MIDI file using music21.

    Args:
        midi_path (str): Path to the MIDI file.

    Returns:
        dict: A dictionary containing calculated features, or None if parsing fails.
              Features: 'pitch_min', 'pitch_max', 'pitch_range',
                        'avg_pitch', 'unique_pitch_classes_count'
    """
    features = {
        'pitch_min': None,
        'pitch_max': None,
        'pitch_range': None,
        'avg_pitch': None,
        'unique_pitch_classes_count': None
    }
    all_pitches = []
    all_pitch_classes = set()

    try:
        # Load the MIDI file
        midi_stream = music21.converter.parse(midi_path)

        notes_and_chords = midi_stream.flat.notes

        if not notes_and_chords:
             print(f"Warning: No notes or chords found in {midi_path}")
             return None

        for element in notes_and_chords:
            if isinstance(element, music21.note.Note):
                pitch_midi = element.pitch.midi # MIDI pitch number (e.g., 60 for C4)
                all_pitches.append(pitch_midi)
                all_pitch_classes.add(element.pitch.pitchClass) # Pitch class (0-11)
            elif isinstance(element, music21.chord.Chord):
                chord_pitches = [p.midi for p in element.pitches]
                all_pitches.extend(chord_pitches)
                # Add pitch classes of all notes in the chord
                for p in element.pitches:
                     all_pitch_classes.add(p.pitchClass)

        if not all_pitches:
             print(f"Warning: Could not extract any pitches from {midi_path}")
             return None # Return None if extraction failed

        # Calculate features
        features['pitch_min'] = int(np.min(all_pitches))
        features['pitch_max'] = int(np.max(all_pitches))
        features['pitch_range'] = features['pitch_max'] - features['pitch_min']
        features['avg_pitch'] = float(np.mean(all_pitches))
        features['unique_pitch_classes_count'] = len(all_pitch_classes)

        return features

    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return None


def run_analysis(base_dir, models, emotions):
    """
    Runs the feature extraction over all specified models and emotions.

    Args:
        base_dir (str): The base directory containing generated MIDI files.
        models (list): A list of model names (subdirectories).
        emotions (list): A list of emotion names (sub-subdirectories).

    Returns:
        pandas.DataFrame: A DataFrame containing features for all processed files.
    """
    results = []

    print(f"Starting analysis in base directory: {base_dir}")

    for model_name in models:
        for emotion_name in emotions:
            current_dir = os.path.join(base_dir, model_name, emotion_name)
            print(f"Processing directory: {current_dir}")

            if not os.path.isdir(current_dir):
                print(f"Warning: Directory not found, skipping.")
                continue

            for filename in os.listdir(current_dir):
                if filename.lower().endswith(('.mid', '.midi')):
                    file_path = os.path.join(current_dir, filename)
                    print(f"Analyzing file: {filename}")
                    features = extract_midi_features(file_path)

                    if features:
                        features['model'] = model_name
                        features['emotion'] = emotion_name
                        features['filename'] = filename
                        results.append(features)
                    else:
                         print(f"Skipped file due to errors or no notes.")

    if not results:
        print("No MIDI files were successfully processed.")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    cols_order = ['model', 'emotion', 'filename', 'pitch_min', 'pitch_max',
                  'pitch_range', 'avg_pitch', 'unique_pitch_classes_count']
    df = df.reindex(columns=[col for col in cols_order if col in df.columns])

    print("\nAnalysis complete.")
    print(f"Successfully processed {len(df)} files.")
    return df


def generate_plots_and_tables(df):
    """
    Generates box plots and summary tables from the feature DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame containing the extracted features.
    """
    if df.empty:
        print("DataFrame is empty, cannot generate plots or tables.")
        return

    print("\nGenerating visualizations and tables...")

    # Calculate mean and std deviation per model and emotion
    summary = df.groupby(['model', 'emotion']).agg(
        Avg_Pitch_Range=('pitch_range', 'mean'),
        Std_Pitch_Range=('pitch_range', 'std'),
        Avg_Avg_Pitch=('avg_pitch', 'mean'),
        Std_Avg_Pitch=('avg_pitch', 'std'),
        Avg_Unique_PCs=('unique_pitch_classes_count', 'mean'),
        Std_Unique_PCs=('unique_pitch_classes_count', 'std'),
        Num_Files=('filename', 'count')
    ).reset_index()

    print("\n--- Summary Statistics Table ---")
    print(summary.round(2).to_markdown(index=False))

    # Features to plot
    features_to_plot = ['pitch_range', 'avg_pitch', 'unique_pitch_classes_count']
    feature_titles = {
        'pitch_range': 'Pitch Range (Max - Min MIDI Note)',
        'avg_pitch': 'Average Pitch (Mean MIDI Note)',
        'unique_pitch_classes_count': 'Number of Unique Pitch Classes (0-11)'
    }

    # Set plot style
    sns.set_theme(style="whitegrid")

    for feature in features_to_plot:
        if feature not in df.columns:
             print(f"Warning: Feature '{feature}' not found in DataFrame, skipping plot.")
             continue

        plt.figure(figsize=(12, 7))
        # Create a box plot comparing models for each emotion
        sns.boxplot(data=df, x='emotion', y=feature, hue='model', palette="Set2")
        plt.title(f'Distribution of {feature_titles.get(feature, feature)} by Emotion and Model')
        plt.xlabel('Target Emotion')
        plt.ylabel(feature_titles.get(feature, feature))
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

        # Save the plot
        plot_filename = f'plot_{feature}_comparison.png'
        plt.savefig(plot_filename)
        print(f"Saved plot: {plot_filename}")
        plt.close()

    print("\nPlots generated successfully.")


# Run the analysis and generate plots/tables
if __name__ == "__main__":
    # Make sure the base directory exists
    if not os.path.isdir(MIDI_BASE_DIR):
         print(f"Error: Base directory '{MIDI_BASE_DIR}' not found.")
         print("Please create the directory structure and place generated MIDI files inside,")
         print("or update the MIDI_BASE_DIR variable in the script.")
    else:
        results_df = run_analysis(MIDI_BASE_DIR, MODELS, EMOTIONS)
        generate_plots_and_tables(results_df)
