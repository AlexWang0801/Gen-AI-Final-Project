import streamlit as st
import os
import random
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import re
import time
import sys

# -----------------------------
# Configuration
# -----------------------------
API_KEY = "Sk-or-v1-ac68cef4655dff8e06543e49c6d2895cd84376416bf266b7b3bb38195dc653ea"  # Replace with your OpenRouter key
LLM_MODEL = "meta-llama/Llama-3-8B-Instruct"  # LLM model to use
MUSIC_MODEL = "facebook/musicgen-small"  # MusicGen model to use locally
DEFAULT_DURATION = 15  # Default music duration in seconds

# Import the complex emotion map here
# (For brevity, we're assuming the complex_emotion_map and emotion_descriptions from previous code are inserted here)
# Complex emotion mapping for MusicGen
complex_emotion_map = {
    # Basic emotions with descriptions
    "joy": {
        "description": "You're feeling joyful and cheerful! 😊",
        "prompt": "upbeat cheerful melody with bright major chords",
        "genre": ["pop", "folk"],
        "instruments": ["piano", "acoustic guitar", "light percussion"],
        "tempo": "medium-fast",
        "tonality": "major"
    },
    "sadness": {
        "description": "Sounds like you're feeling a bit down. 😢",
        "prompt": "melancholic emotional piece with minor progressions",
        "genre": ["classical", "ambient"],
        "instruments": ["piano", "strings", "soft synth"],
        "tempo": "slow",
        "tonality": "minor"
    },
    "anger": {
        "description": "You're expressing anger or frustration. 😠",
        "prompt": "intense forceful composition with powerful dynamics",
        "genre": ["rock", "orchestral"],
        "instruments": ["distorted guitar", "drums", "brass", "strings"],
        "tempo": "fast, driving",
        "tonality": "minor, dissonant"
    },
    "gratitude": {
        "description": "You're feeling thankful and appreciative. 🙏",
        "prompt": "warm heartfelt composition with sincere qualities",
        "genre": ["folk", "acoustic"],
        "instruments": ["acoustic guitar", "piano", "soft strings"],
        "tempo": "moderate, flowing",
        "tonality": "major, resolved"
    },
    "love": {
        "description": "Your message shows love and affection. ❤️",
        "prompt": "romantic tender composition with intimate qualities",
        "genre": ["classical", "jazz ballad"],
        "instruments": ["piano", "strings", "acoustic guitar"],
        "tempo": "slow to moderate",
        "tonality": "lush major"
    },
    "fear": {
        "description": "There's a sense of fear or anxiety. 😨",
        "prompt": "tense suspenseful piece with unsettling elements",
        "genre": ["horror score", "experimental"],
        "instruments": ["strings", "prepared piano", "electronics", "percussion"],
        "tempo": "variable, unpredictable",
        "tonality": "atonal, dissonant"
    },
    "surprise": {
        "description": "That caught you off guard! 😲",
        "prompt": "unexpected playful composition with sudden shifts",
        "genre": ["quirky", "experimental"],
        "instruments": ["pizzicato strings", "woodwinds", "unusual percussion"],
        "tempo": "unpredictable",
        "tonality": "shifting, unexpected progressions"
    },
    "neutral": {
        "description": "A calm, neutral mood. 😐",
        "prompt": "balanced composition with moderate emotional qualities",
        "genre": ["ambient", "background"],
        "instruments": ["piano", "soft synth", "light strings"],
        "tempo": "moderate, steady",
        "tonality": "balanced major/minor"
    },
    "peaceful": {
        "description": "You're calm and at ease. 🌿",
        "prompt": "calming gentle composition with soothing qualities",
        "genre": ["ambient", "meditation"],
        "instruments": ["soft piano", "gentle pads", "nature sounds"],
        "tempo": "slow, gentle",
        "tonality": "major, consonant"
    },
    "tense": {
        "description": "There's tension or unease. 😬",
        "prompt": "unsettling composition with building tension",
        "genre": ["thriller score", "contemporary"],
        "instruments": ["strings", "percussion", "electronics"],
        "tempo": "building, insistent",
        "tonality": "dissonant, unresolved"
    },
    
    # Complex emotions
    "admiration": {
        "description": "You're expressing deep respect or wonder for something or someone.",
        "prompt": "noble uplifting composition with reverent qualities",
        "genre": ["orchestral", "cinematic"],
        "instruments": ["strings", "horns", "piano"],
        "tempo": "moderate, dignified",
        "tonality": "major, resolved"
    },
    "amusement": {
        "description": "You find something funny or entertaining.",
        "prompt": "playful lighthearted melody with whimsical elements",
        "genre": ["light jazz", "quirky folk"],
        "instruments": ["pizzicato strings", "woodwinds", "playful percussion"],
        "tempo": "medium, bouncy",
        "tonality": "major, light"
    },
    "annoyance": {
        "description": "Something is bothering or irritating you.",
        "prompt": "slightly agitated composition with persistent patterns",
        "genre": ["contemporary", "minimalist"],
        "instruments": ["piano", "percussion", "strings"],
        "tempo": "medium, insistent",
        "tonality": "dissonant, unresolved"
    },
    "approval": {
        "description": "You're expressing agreement or endorsement.",
        "prompt": "affirming positive composition with confident progression",
        "genre": ["contemporary", "pop"],
        "instruments": ["piano", "guitar", "light percussion"],
        "tempo": "moderate, confident",
        "tonality": "major, resolved"
    },
    "caring": {
        "description": "You're showing concern or tenderness for others.",
        "prompt": "gentle nurturing composition with warm tones",
        "genre": ["folk", "soft classical"],
        "instruments": ["acoustic guitar", "piano", "soft strings"],
        "tempo": "slow to moderate",
        "tonality": "warm major"
    },
    "confusion": {
        "description": "You're feeling puzzled or uncertain about something.",
        "prompt": "wandering unpredictable composition with changing patterns",
        "genre": ["experimental", "contemporary"],
        "instruments": ["piano", "electronics", "woodwinds"],
        "tempo": "irregular, shifting",
        "tonality": "modal, ambiguous"
    },
    "curiosity": {
        "description": "You're expressing interest and a desire to learn more.",
        "prompt": "inquisitive playful composition with exploratory melody",
        "genre": ["contemporary", "chamber music"],
        "instruments": ["piano", "clarinet", "light percussion"],
        "tempo": "moderate, questioning",
        "tonality": "modal, open-ended"
    },
    "desire": {
        "description": "You're expressing a strong wish or craving for something.",
        "prompt": "yearning expressive composition with passionate qualities",
        "genre": ["contemporary classical", "cinematic"],
        "instruments": ["cello", "piano", "strings"],
        "tempo": "moderate, flowing",
        "tonality": "rich minor with tension"
    },
    "disappointment": {
        "description": "Your expectations weren't met, leaving you feeling letdown.",
        "prompt": "descending melancholic piece with resigned qualities",
        "genre": ["ambient", "soft piano"],
        "instruments": ["piano", "soft synth", "subtle strings"],
        "tempo": "slow, falling",
        "tonality": "minor with descending progression"
    },
    "disapproval": {
        "description": "You're expressing disagreement or negative judgment.",
        "prompt": "stern composition with critical qualities and tension",
        "genre": ["contemporary", "dissonant"],
        "instruments": ["piano", "low strings", "percussion"],
        "tempo": "moderate, deliberate",
        "tonality": "minor, dissonant"
    },
    "disgust": {
        "description": "You're expressing strong aversion or revulsion.",
        "prompt": "unsettling dissonant composition with jarring elements",
        "genre": ["experimental", "atonal"],
        "instruments": ["prepared piano", "electronics", "distorted sounds"],
        "tempo": "irregular, unsettling",
        "tonality": "dissonant, atonal"
    },
    "embarrassment": {
        "description": "You're feeling self-conscious or uncomfortable about something.",
        "prompt": "awkward hesitant composition with uncertain progression",
        "genre": ["quirky", "minimal"],
        "instruments": ["muted piano", "pizzicato strings", "light percussion"],
        "tempo": "hesitant, uneven",
        "tonality": "shifting, unresolved"
    },
    "excitement": {
        "description": "You're feeling enthusiastic and eager about something.",
        "prompt": "energetic vibrant composition with building anticipation",
        "genre": ["electronic", "cinematic"],
        "instruments": ["synth", "drums", "strings"],
        "tempo": "fast, building",
        "tonality": "bright major with dynamic progression"
    },
    "grief": {
        "description": "You're experiencing deep sorrow, typically after loss.",
        "prompt": "deeply sorrowful composition with profound emotional weight",
        "genre": ["classical", "ambient"],
        "instruments": ["solo cello", "piano", "subtle strings"],
        "tempo": "very slow, heavy",
        "tonality": "deep minor, somber"
    },
    "nervousness": {
        "description": "You're feeling anxious or worried about something.",
        "prompt": "uneasy composition with fluttering patterns and tension",
        "genre": ["contemporary", "minimal"],
        "instruments": ["tremolo strings", "piano", "light percussion"],
        "tempo": "moderate with irregular accents",
        "tonality": "unstable, shifting"
    },
    "optimism": {
        "description": "You're feeling positive and hopeful about the future.",
        "prompt": "bright forward-looking composition with uplifting progression",
        "genre": ["contemporary", "cinematic"],
        "instruments": ["piano", "strings", "light percussion"],
        "tempo": "moderate, forward-moving",
        "tonality": "bright major with rising phrases"
    },
    "pride": {
        "description": "You're feeling satisfaction from achievement or recognition.",
        "prompt": "dignified triumphant composition with noble qualities",
        "genre": ["orchestral", "cinematic"],
        "instruments": ["brass", "strings", "timpani"],
        "tempo": "moderate, stately",
        "tonality": "strong major, resolving"
    },
    "realization": {
        "description": "You've just understood or become aware of something important.",
        "prompt": "revelatory composition with dawning clarity",
        "genre": ["contemporary", "cinematic"],
        "instruments": ["piano", "strings", "subtle electronics"],
        "tempo": "building from slow to moderate",
        "tonality": "progression from ambiguous to clear resolution"
    },
    "relief": {
        "description": "You're feeling reassured after anxiety or distress has passed.",
        "prompt": "releasing composition with tension resolving into calm",
        "genre": ["ambient", "contemporary"],
        "instruments": ["piano", "strings", "gentle synth"],
        "tempo": "slow to moderate, easing",
        "tonality": "progression from tension to resolution"
    },
    "remorse": {
        "description": "You're feeling deep regret for something you've done.",
        "prompt": "reflective sorrowful composition with introspective qualities",
        "genre": ["classical", "ambient"],
        "instruments": ["piano", "cello", "soft winds"],
        "tempo": "slow, thoughtful",
        "tonality": "minor, contemplative"
    }
}

# Emotion descriptions for UI display
emotion_descriptions = {
    "joy": "You're feeling joyful and cheerful! 😊",
    "sadness": "Sounds like you're feeling a bit down. 😢",
    "anger": "You're expressing anger or frustration. 😠",
    "gratitude": "You're feeling thankful and appreciative. 🙏",
    "love": "Your message shows love and affection. ❤️",
    "fear": "There's a sense of fear or anxiety. 😨",
    "surprise": "That caught you off guard! 😲",
    "neutral": "A calm, neutral mood. 😐",
    "peaceful": "You're calm and at ease. 🌿",
    "tense": "There's tension or unease. 😬",
    "admiration": "You're expressing deep respect or wonder for something or someone.",
    "amusement": "You find something funny or entertaining.",
    "annoyance": "Something is bothering or irritating you.",
    "approval": "You're expressing agreement or endorsement.",
    "caring": "You're showing concern or tenderness for others.",
    "confusion": "You're feeling puzzled or uncertain about something.",
    "curiosity": "You're expressing interest and a desire to learn more.",
    "desire": "You're expressing a strong wish or craving for something.",
    "disappointment": "Your expectations weren't met, leaving you feeling letdown.",
    "disapproval": "You're expressing disagreement or negative judgment.",
    "disgust": "You're expressing strong aversion or revulsion.",
    "embarrassment": "You're feeling self-conscious or uncomfortable about something.",
    "excitement": "You're feeling enthusiastic and eager about something.",
    "grief": "You're experiencing deep sorrow, typically after loss.",
    "nervousness": "You're feeling anxious or worried about something.",
    "optimism": "You're feeling positive and hopeful about the future.",
    "pride": "You're feeling satisfaction from achievement or recognition.",
    "realization": "You've just understood or become aware of something important.",
    "relief": "You're feeling reassured after anxiety or distress has passed.",
    "remorse": "You're feeling deep regret for something you've done."
}

# List of all available emotions for Llama 3 to choose from
all_emotions = list(complex_emotion_map.keys())
# --- Page config must come first ---
st.set_page_config(
    page_title="🎵 Emotion Music Generator (API Version)", 
    layout="wide",
    initial_sidebar_state="expanded"
)

def detect_complex_emotion_with_api(text):
    """
    Use OpenRouter API to detect complex emotions in text with Llama 3.
    Includes better handling of ambiguous emotions.
    
    Args:
        text (str): The input text to analyze
        
    Returns:
        dict: Containing primary emotion, secondary emotion, intensity, and context
    """
    print(f"Analyzing text: '{text[:50]}...'", file=sys.stderr)
    
    # Get list of all available emotions, including complex/ambiguous ones
    available_emotions = list(complex_emotion_map.keys())
    
    # Separate emotions into basic and complex for better prompting
    basic_emotions = ["joy", "sadness", "anger", "gratitude", "love", "fear", "surprise", "neutral", "peaceful", "tense"]
    complex_emotions = [e for e in available_emotions if e not in basic_emotions]
    
    # Create a well-structured prompt that encourages nuanced emotion detection
    prompt = f"""Analyze the emotional content of the following text with NUANCE and DEPTH.

Text: "{text}"

IMPORTANT: People often experience complex, ambiguous, or mixed emotions rather than just basic emotions.

Consider these emotion categories carefully:

BASIC EMOTIONS: {", ".join(basic_emotions)}

COMPLEX/AMBIGUOUS EMOTIONS: {", ".join(complex_emotions)}

Instructions:
1. First, identify the PRIMARY emotion that BEST captures the emotional state in the text.
2. Then, identify any SECONDARY emotion that adds nuance to the emotional state.
3. Rate the emotional INTENSITY on a scale of 0.0 to 1.0, where 0.0 is very subtle and 1.0 is extremely intense.
4. Provide a brief CONTEXT explaining your analysis.

Always prefer a specific complex emotion over a general basic one if it better captures the emotional state.

Return your analysis in this exact format:
Primary: [primary emotion]
Secondary: [secondary emotion or "none" if not applicable]
Intensity: [number between 0.0-1.0]
Context: [brief explanation of your analysis]"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 300
    }
    
    try:
        with st.spinner("Contacting OpenRouter API..."):
            print("Sending request to OpenRouter API...", file=sys.stderr)
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            
            response_json = response.json()
            assistant_message = response_json["choices"][0]["message"]["content"]
            
            print(f"Received response from API: {assistant_message[:100]}...", file=sys.stderr)
            
            # For debugging
            st.session_state.last_api_response = assistant_message
            
            # Parse the structured response
            result = {
                "primary": "neutral",
                "secondary": None,
                "intensity": 0.5,
                "context": ""
            }
            
            try:
                # Extract primary emotion
                if "Primary:" in assistant_message:
                    primary_line = assistant_message.split("Primary:")[1].split("\n")[0].strip()
                    
                    # Check for exact matches first
                    for emotion in available_emotions:
                        if emotion.lower() == primary_line.lower():
                            result["primary"] = emotion
                            break
                    
                    # If no exact match, look for partial matches
                    if result["primary"] == "neutral":
                        for emotion in available_emotions:
                            if emotion.lower() in primary_line.lower():
                                result["primary"] = emotion
                                break
                
                # Extract secondary emotion
                if "Secondary:" in assistant_message:
                    secondary_line = assistant_message.split("Secondary:")[1].split("\n")[0].strip()
                    if "none" not in secondary_line.lower() and "n/a" not in secondary_line.lower():
                        # Check for exact matches first
                        for emotion in available_emotions:
                            if emotion.lower() == secondary_line.lower():
                                result["secondary"] = emotion
                                break
                        
                        # If no exact match, look for partial matches
                        if result["secondary"] is None:
                            for emotion in available_emotions:
                                if emotion.lower() in secondary_line.lower():
                                    result["secondary"] = emotion
                                    break
                
                # Extract intensity
                if "Intensity:" in assistant_message:
                    intensity_text = assistant_message.split("Intensity:")[1].split("\n")[0].strip()
                    # Extract numeric value using regex
                    intensity_match = re.search(r'(\d+\.\d+|\d+)', intensity_text)
                    if intensity_match:
                        intensity = float(intensity_match.group(1))
                        result["intensity"] = max(0.0, min(1.0, intensity))  # Clamp to 0.0-1.0
                
                # Extract context
                if "Context:" in assistant_message:
                    result["context"] = assistant_message.split("Context:")[1].strip()
                
                print(f"Detected emotions: Primary={result['primary']}, Secondary={result['secondary']}, Intensity={result['intensity']}", file=sys.stderr)
            
            except Exception as e:
                st.warning(f"Error parsing response: {str(e)}")
                st.caption(f"Raw response: {assistant_message}")
                print(f"Error parsing response: {str(e)}", file=sys.stderr)
            
            return result
            
    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        st.error(error_msg)
        print(error_msg, file=sys.stderr)
        return {"primary": "neutral", "secondary": None, "intensity": 0.5, "context": f"Error: {str(e)}"}

def blend_emotions_for_music(primary, secondary=None, blend_ratio=0.7):
    """
    Create a more sophisticated blending of emotions for music generation.
    
    Args:
        primary (str): Primary emotion
        secondary (str): Secondary emotion (optional)
        blend_ratio (float): Weight of primary emotion (0.0-1.0)
        
    Returns:
        dict: Blended emotion characteristics
    """
    # Get primary emotion data
    primary_data = complex_emotion_map.get(primary, complex_emotion_map["neutral"])
    
    # If no secondary emotion, just return primary
    if not secondary:
        return primary_data
    
    # Get secondary emotion data
    secondary_data = complex_emotion_map.get(secondary, complex_emotion_map["neutral"])
    
    # Create blended emotion data
    blended_data = {
        "prompt": f"A blend of {primary} and {secondary}: {primary_data['prompt']} with elements of {secondary_data['prompt']}",
        "genre": list(set(primary_data["genre"] + secondary_data["genre"])),  # Combine unique genres
        "instruments": list(set(primary_data["instruments"] + secondary_data["instruments"])),  # Combine unique instruments
        "tempo": f"Primarily {primary_data['tempo']} with moments of {secondary_data['tempo']}",
        "tonality": f"Primarily {primary_data['tonality']} with elements of {secondary_data['tonality']}"
    }
    
    print(f"Blended emotions: {primary} + {secondary}", file=sys.stderr)
    return blended_data

def generate_musicgen_prompt(emotion_result, text_input):
    """
    Generate a detailed prompt for MusicGen based on emotion analysis results.
    
    Args:
        emotion_result (dict): The emotion analysis result
        text_input (str): The original user input text
        
    Returns:
        str: A detailed prompt for MusicGen
    """
    primary_emotion = emotion_result["primary"]
    secondary_emotion = emotion_result["secondary"]
    emotional_intensity = emotion_result["intensity"]
    
    # Get emotion data - either single or blended
    if secondary_emotion:
        emotion_data = blend_emotions_for_music(primary_emotion, secondary_emotion)
    else:
        # Use the complex emotion map if available
        emotion_data = complex_emotion_map.get(primary_emotion, complex_emotion_map["neutral"])
    
    # Extract emotion characteristics
    base_prompt = emotion_data["prompt"]
    genres = emotion_data["genre"]
    instruments = emotion_data["instruments"]
    tempo = emotion_data["tempo"]
    tonality = emotion_data["tonality"]
    
    # Select random elements for variety
    genre = random.choice(genres) if len(genres) > 0 else "ambient"
    selected_instruments = random.sample(instruments, min(3, len(instruments))) if len(instruments) > 0 else ["piano"]
    
    # Adjust intensity based on parameter
    intensity_phrases = {
        0.25: "subtle, gentle, understated",
        0.5: "moderate, balanced, clear",
        0.75: "pronounced, expressive, definite",
        1.0: "intense, powerful, dramatic"
    }
    
    # Find closest intensity level
    intensity_levels = sorted(intensity_phrases.keys())
    closest_intensity = min(intensity_levels, key=lambda x: abs(x - emotional_intensity))
    intensity_phrase = intensity_phrases[closest_intensity]
    
    # Extract key words from original text
    words = text_input.lower().split()
    descriptive_words = [word for word in words if len(word) > 4 and word not in 
                         ["about", "above", "across", "after", "against", "along", "among", "around"]]
    text_context = ""
    if descriptive_words:
        selected_words = random.sample(descriptive_words, min(2, len(descriptive_words)))
        text_context = f" with elements of {' and '.join(selected_words)}"
    
    # Build the prompt
    prompt = f"{genre} music with {', '.join(selected_instruments)}. {base_prompt}{text_context}. "
    prompt += f"{intensity_phrase} emotional qualities. {tempo} tempo in {tonality} tonality."
    
    print(f"Generated MusicGen prompt: {prompt}", file=sys.stderr)
    return prompt

# Initialize MusicGen locally
@st.cache_resource
def load_musicgen_model():
    print("Starting to load MusicGen model...", file=sys.stderr)
    try:
        model = MusicGen.get_pretrained(MUSIC_MODEL)
        print(f"MusicGen model loaded successfully: {MUSIC_MODEL}", file=sys.stderr)
        print(f"Model parameters: {model.lm.config.vocab_size} vocab size, {model.lm.config.d_model} dimensions", file=sys.stderr)
        return model
    except Exception as e:
        print(f"Error loading MusicGen model: {str(e)}", file=sys.stderr)
        raise e

# Function to display emotion analysis results
def display_emotion_analysis(emotion_result):
    """Display the emotion analysis results in the UI"""
    primary_emotion = emotion_result["primary"]
    secondary_emotion = emotion_result["secondary"]
    intensity = emotion_result["intensity"]
    context = emotion_result["context"]
    
    # Get descriptions for display
    primary_desc = complex_emotion_map.get(primary_emotion, {}).get("description", primary_emotion.capitalize())
    
    # Display emotional analysis results
    st.markdown(f"**Primary Emotion**: {primary_emotion.capitalize()}")
    st.markdown(f"**Description**: {primary_desc}")
    
    if secondary_emotion:
        secondary_desc = complex_emotion_map.get(secondary_emotion, {}).get("description", secondary_emotion.capitalize())
        st.markdown(f"**Secondary Emotion**: {secondary_emotion.capitalize()}")
        st.markdown(f"**Secondary Description**: {secondary_desc}")
        
    st.markdown(f"**Emotional Intensity**: {intensity:.2f}")
    st.markdown(f"**Analysis Context**: {context}")
    
    # Create emotion visualization
    fig, ax = plt.subplots(figsize=(4, 3))
    
    if secondary_emotion:
        emotions = [primary_emotion, secondary_emotion]
        values = [intensity, max(0.2, intensity * 0.6)]  # Scale secondary emotion
        colors = ["#FF9999", "#9999FF"]
    else:
        emotions = [primary_emotion]
        values = [intensity]
        colors = ["#FF9999"]
    
    # Create simple bar chart
    ax.bar(emotions, values, color=colors)
    ax.set_ylabel("Intensity")
    ax.set_title("Emotional Analysis")
    ax.set_ylim(0, 1.0)
    st.pyplot(fig)
    
    # Show the raw API response in a collapsible section (for debugging)
    if 'last_api_response' in st.session_state and not st.session_state.get('manual_override', False):
        with st.expander("Show Raw API Response"):
            st.text(st.session_state.last_api_response)

# Function to generate and display music
def generate_and_display_music(emotion_result, text_input, duration):
    """Generate and display music based on emotion analysis"""
    if 'musicgen_loaded' not in st.session_state:
        st.warning("⚠️ Please load the MusicGen model first using the button in the sidebar")
        return
    
    with st.spinner("Creating emotionally resonant music..."):
        # Create outputs directory if it doesn't exist
        os.makedirs("outputs", exist_ok=True)
        
        # Generate music prompt
        detailed_prompt = generate_musicgen_prompt(emotion_result, text_input)
        
        st.info(f"**MusicGen Prompt:**\n{detailed_prompt}")
        
        # Generate music
        musicgen = st.session_state.musicgen
        musicgen.set_generation_params(duration=duration)
        
        print(f"Generating music with duration: {duration} seconds...", file=sys.stderr)
        output = musicgen.generate([detailed_prompt])
        print("Music generation complete!", file=sys.stderr)
        
        # Create filename based on emotions
        primary_emotion = emotion_result["primary"]
        secondary_emotion = emotion_result["secondary"]
        intensity = emotion_result["intensity"]
        
        emotion_str = f"{primary_emotion}"
        if secondary_emotion:
            emotion_str += f"_{secondary_emotion}"
        output_path = f"outputs/{emotion_str}_intensity{int(intensity*10)}.wav"
        
        # Save and display
        print(f"Saving music to: {output_path}", file=sys.stderr)
        audio_write(output_path[:-4], output[0].cpu(), sample_rate=32000)
        st.audio(output_path)
        
        # Download button
        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Music", f, file_name=os.path.basename(output_path))
        
    # Show musical characteristics
    with st.expander("🎼 Musical Characteristics"):
        st.markdown("### How the emotion was translated to music")
        
        if secondary_emotion:
            st.markdown(f"### Blended emotion: {primary_emotion} + {secondary_emotion}")
            
            # Get both emotion data
            primary_data = complex_emotion_map.get(primary_emotion, complex_emotion_map["neutral"])
            secondary_data = complex_emotion_map.get(secondary_emotion, complex_emotion_map["neutral"])
            
            # Create a two-column comparison
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Primary Emotion ({primary_emotion})**")
                st.markdown(f"**Base Prompt**: {primary_data['prompt']}")
                st.markdown(f"**Genres**: {', '.join(primary_data['genre'])}")
                st.markdown(f"**Instruments**: {', '.join(primary_data['instruments'])}")
                st.markdown(f"**Tempo**: {primary_data['tempo']}")
                st.markdown(f"**Tonality**: {primary_data['tonality']}")
            
            with col2:
                st.markdown(f"**Secondary Emotion ({secondary_emotion})**")
                st.markdown(f"**Base Prompt**: {secondary_data['prompt']}")
                st.markdown(f"**Genres**: {', '.join(secondary_data['genre'])}")
                st.markdown(f"**Instruments**: {', '.join(secondary_data['instruments'])}")
                st.markdown(f"**Tempo**: {secondary_data['tempo']}")
                st.markdown(f"**Tonality**: {secondary_data['tonality']}")
        else:
            # Get emotion data for a single emotion
            emotion_data = complex_emotion_map.get(primary_emotion, complex_emotion_map["neutral"])
            
            # Display in a nice format
            st.markdown(f"### {primary_emotion.capitalize()}")
            st.markdown(f"**Base Prompt**: {emotion_data['prompt']}")
            st.markdown(f"**Genres**: {', '.join(emotion_data['genre'])}")
            st.markdown(f"**Instruments**: {', '.join(emotion_data['instruments'])}")
            st.markdown(f"**Tempo**: {emotion_data['tempo']}")
            st.markdown(f"**Tonality**: {emotion_data['tonality']}")

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Only show API key field if not already configured at the top
    if not API_KEY or API_KEY == "Sk-or-v1-ac68cef4655dff8e06543e49c6d2895cd84376416bf266b7b3bb38195dc653ea":
        api_key = st.text_input("OpenRouter API Key", value="", type="password")
        if api_key:
            API_KEY = api_key
    
    duration = st.slider("Music Duration (seconds)", min_value=5, max_value=30, value=DEFAULT_DURATION, step=5)
    
    # Advanced emotion settings
    st.subheader("Emotion Processing")
    manual_override = st.checkbox("Manual Emotion Override", value=False)
    st.session_state.manual_override = manual_override
    
    # Only show these if manual override is checked
    if manual_override:
        all_emotions = sorted(list(complex_emotion_map.keys()))
        selected_emotion = st.selectbox("Primary Emotion", all_emotions)
        secondary_emotion = st.selectbox("Secondary Emotion (optional)", ["None"] + all_emotions)
        secondary_emotion = None if secondary_emotion == "None" else secondary_emotion
        emotion_intensity = st.slider("Emotional Intensity", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    # Separate button for loading MusicGen
    st.subheader("Music Generation")
    load_musicgen_button = st.button("Load MusicGen Model")
    musicgen_status = st.empty()
    
    if load_musicgen_button or 'musicgen_loaded' in st.session_state:
        with musicgen_status:
            with st.spinner("Loading MusicGen model... this may take a minute"):
                try:
                    musicgen = load_musicgen_model()
                    st.session_state.musicgen_loaded = True
                    st.session_state.musicgen = musicgen
                    st.success("✅ MusicGen model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading MusicGen model: {str(e)}")

# Main application UI
st.title("🎵 Emotion Music Generator")
st.markdown("""
This application generates music that matches the emotional nuance of your text, including complex
and ambiguous emotions like nostalgia, anticipation, or melancholy.

### How it works:
1. 🦙 **Llama** analyzes your text for emotions
2. 🎭 The system maps your emotional state to musical characteristics
3. 🎶 **MusicGen** generates a music piece based on the emotional qualities
""")

# Text input area
user_input = st.text_area(
    "💬 Express yourself - what are you feeling?",
    placeholder="e.g., I feel a strange mixture of nostalgia and hope as I look through these old photographs...",
    height=150
)

# Set user_input from session state if it exists (for examples)
if "user_input" in st.session_state:
    user_input = st.session_state.user_input
    # Clear it so it doesn't persist
    del st.session_state.user_input

# STEP 1: Analyze Emotion Button
if st.button("🔍 Analyze Emotion") and user_input:
    # Check if API key is set
    if not API_KEY or API_KEY == "Sk-or-v1-ac68cef4655dff8e06543e49c6d2895cd84376416bf266b7b3bb38195dc653ea":
        st.error("Please set your OpenRouter API key in the configuration or sidebar")
        st.stop()
    
    # Get emotion analysis
    with st.spinner("Analyzing emotions..."):
        if manual_override:
            # Use manually selected emotions
            emotion_result = {
                "primary": selected_emotion,
                "secondary": secondary_emotion,
                "intensity": emotion_intensity,
                "context": "Manually selected emotions"
            }
        else:
            # Get emotion analysis from API
            emotion_result = detect_complex_emotion_with_api(user_input)
    
    # Store the emotion result in session state for later use
    st.session_state.emotion_result = emotion_result
    st.session_state.analyzed_text = user_input
    
    # Display emotion analysis
    display_emotion_analysis(emotion_result)
    
    # Show a message to load MusicGen if not loaded
    if 'musicgen_loaded' not in st.session_state:
        st.info("👈 Now load the MusicGen model from the sidebar to generate music based on this emotion")
    else:
        # Show generate music button
        if st.button("🎵 Generate Music Based on Emotion"):
            generate_and_display_music(emotion_result, user_input, duration)

# STEP 2: Generate Music Button (only show if emotion has been analyzed)
elif 'emotion_result' in st.session_state and 'musicgen_loaded' in st.session_state:
    # Display the previously analyzed emotion
    st.subheader("🔍 Previously Analyzed Emotion")
    display_emotion_analysis(st.session_state.emotion_result)
    
    # Generate music button
    if st.button("🎵 Generate Music Based on Emotion"):
        generate_and_display_music(
            st.session_state.emotion_result, 
            st.session_state.analyzed_text, 
            duration
        )

# Add examples at the bottom
with st.expander("📝 Example Texts to Try"):
    examples = [
        "I feel a bittersweet nostalgia as I look through old photographs, remembering good times while also feeling the passage of time.",
        "I'm filled with anticipation and excitement about my upcoming journey, though there's a hint of nervousness too.",
        "I feel completely at peace watching the sunset by the ocean, with a gentle breeze and the sound of waves.",
        "I'm experiencing a complex mix of pride and melancholy as my child leaves for college - happy for their future but sad about the change.",
        "I feel conflicted about my career decision - part of me is excited for new opportunities while another part fears leaving my comfort zone.",
        "I'm filled with awe looking at the night sky, feeling small yet connected to something vast and beautiful."
    ]
    
    # Create a grid of example buttons
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state["user_input"] = example
                st.experimental_rerun()

if __name__ == "__main__":
    # The app is already running if this script is executed directly
    pass