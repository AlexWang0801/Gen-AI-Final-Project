import streamlit as st
from transformers import pipeline
import scipy.io.wavfile as wav

# Initialize the Streamlit interface
st.title("🎵 AI Music Generator")
st.subheader("Enter your emotions and we’ll generate music to match them!")

# Load the emotion analysis model (using Hugging Face's BERT pre-trained model)
@st.cache_resource
def load_emotion_model():
    return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

emotion_model = load_emotion_model()

# Load the MusicGen-Small model
@st.cache_resource
def load_musicgen_model():
    return pipeline("text-to-audio", model="facebook/musicgen-small")

musicgen_synthesizer = load_musicgen_model()

# User input text
user_input = st.text_area("Describe your emotions (enter a sentence)", "Today I feel very happy!")

# Process the user input
if st.button("Analyze Emotion"):
    result = emotion_model(user_input)
    detected_emotion = result[0]['label']
    
    # Display the detected emotion
    st.write(f"🎭 **Detected Emotion**: {detected_emotion}")

    # Choose music style based on emotion
    emotion_map = {
        "joy": "lo-fi music with a soothing melody",
        "sadness": "sad piano music",
        "anger": "intense electronic music",
        "calm": "calm ambient music",
    }
    
    # Get the corresponding music style
    music_prompt = emotion_map.get(detected_emotion, "neutral music")

    # Generate music using MusicGen-Small
    music = musicgen_synthesizer(music_prompt, forward_params={"do_sample": True})

    # Save the generated audio as a file
    wav.write("generated_music.wav", rate=music["sampling_rate"], data=music["audio"])

    # Display the music
    st.audio("generated_music.wav", format="audio/wav")
    st.write(f"🎼 **Playing music that matches the emotion: {detected_emotion}** 🎶")
