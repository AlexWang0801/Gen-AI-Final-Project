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
from emotion import complex_emotion_map
 
# -----------------------------
# Configuration
# -----------------------------
# Use your API key directly here
API_KEY = "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"
LLM_MODEL = "meta-llama/Llama-3-8B-Instruct"  # LLM model to use
MUSIC_MODEL = "facebook/musicgen-small"  # MusicGen model to use locally
DEFAULT_DURATION = 15  # Default music duration in seconds

# List of all available emotions for Llama 3 to choose from
all_emotions = sorted(list(complex_emotion_map.keys()))

# --- Page config must come first ---
st.set_page_config(
    page_title="🎵 Emotion Music Generator", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'manual_emotions_applied' not in st.session_state:
    st.session_state.manual_emotions_applied = False
    
if 'emotions_analyzed' not in st.session_state:
    st.session_state.emotions_analyzed = False

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
    
    # Extract key words from original text if available
    text_context = ""
    if text_input and text_input != "Custom emotion selection":
        words = text_input.lower().split()
        descriptive_words = [word for word in words if len(word) > 4 and word not in 
                             ["about", "above", "across", "after", "against", "along", "among", "around"]]
        if descriptive_words:
            selected_words = random.sample(descriptive_words, min(2, len(descriptive_words)))
            text_context = f" with elements of {' and '.join(selected_words)}"
    
    # Build the prompt
    prompt = f"{genre} music with {', '.join(selected_instruments)}. {base_prompt}{text_context}. "
    prompt += f"{intensity_phrase} emotional qualities. {tempo} tempo in {tonality} tonality."
    
    print(f"Generated MusicGen prompt: {prompt}", file=sys.stderr)
    return prompt

# Initialize MusicGen locally - FIXED VERSION
@st.cache_resource
def load_musicgen_model():
    print("Starting to load MusicGen model...", file=sys.stderr)
    try:
        # Load model with more explicit error handling
        model = MusicGen.get_pretrained(MUSIC_MODEL)
        print(f"MusicGen model loaded successfully: {MUSIC_MODEL}", file=sys.stderr)
        
        # Instead of trying to access properties that might not exist,
        # just confirm the model has the needed generation capabilities
        if hasattr(model, 'generate') and callable(model.generate):
            print("Model has generation capabilities", file=sys.stderr)
        else:
            print("Warning: Model might not have generation methods", file=sys.stderr)
            
        return model
    except Exception as e:
        print(f"Error loading MusicGen model: {str(e)}", file=sys.stderr)
        # Re-raise exception so it's caught by the caller
        raise e

# Function to display emotion analysis results with intensity bars
def display_emotion_analysis(emotion_result):
    """Display the emotion analysis results in the UI with descriptions and intensity bars for both emotions"""
    primary_emotion = emotion_result["primary"]
    secondary_emotion = emotion_result["secondary"]
    intensity = emotion_result["intensity"]
    
    # Get descriptions for display
    primary_desc = complex_emotion_map.get(primary_emotion, {}).get("description", primary_emotion.capitalize())
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Primary emotion in first column
    with col1:
        st.markdown("### Primary Emotion")
        st.markdown(f"**{primary_emotion.capitalize()}** (Intensity: {intensity:.2f})")
        st.markdown(f"*{primary_desc}*")
        
        # Create intensity bar for primary emotion
        st.progress(intensity)
    
    # Secondary emotion in second column (if exists)
    with col2:
        if secondary_emotion:
            secondary_desc = complex_emotion_map.get(secondary_emotion, {}).get("description", secondary_emotion.capitalize())
            # Calculate secondary intensity (typically lower than primary)
            secondary_intensity = max(0.2, intensity * 0.6)  # Scale secondary emotion intensity
            
            st.markdown("### Secondary Emotion")
            st.markdown(f"**{secondary_emotion.capitalize()}** (Intensity: {secondary_intensity:.2f})")
            st.markdown(f"*{secondary_desc}*")
            
            # Create intensity bar for secondary emotion
            st.progress(secondary_intensity)
        else:
            st.markdown("### Secondary Emotion")
            st.markdown("*No secondary emotion detected*")
            
            # Empty/zero intensity bar for visual consistency
            st.progress(0.0)

# Function to generate and play music with a simplified UI layout
def generate_and_display_music(emotion_result, text_input, duration):
    """Generate and display music based on emotion analysis"""
    if 'musicgen_loaded' not in st.session_state:
        st.warning("⚠️ Please load the MusicGen model first using the button in the sidebar")
        return
    
    with st.spinner("🎼 Generating high-quality music with MusicGen..."):
        # Generate music prompt
        detailed_prompt = generate_musicgen_prompt(emotion_result, text_input)
        
        try:
            # Generate music with more error handling
            musicgen = st.session_state.musicgen
            
            print(f"Generating music with duration: {duration} seconds...", file=sys.stderr)
            
            # Set generation parameters
            try:
                musicgen.set_generation_params(duration=duration)
            except Exception as e:
                print(f"Error setting generation params: {str(e)}", file=sys.stderr)
            
            # Generate the music
            output = musicgen.generate([detailed_prompt])
            print("Music generation complete!", file=sys.stderr)
            
            # Display audio directly in UI
            import numpy as np
            
            # Get the audio as numpy array
            audio_array = output[0].cpu().numpy()
            
            # Store the audio data in session state (to keep it available)
            st.session_state['generated_audio'] = audio_array
            
            # Display success message
            st.success("✅ Your music is ready! Play it below:")
            
            # Create a container for the audio player (makes it more prominent)
            audio_container = st.container()
            with audio_container:
                # Display using built-in audio component with custom sampling rate
                st.audio(audio_array, sample_rate=32000)
            
            # Display the music prompt details in expander
            with st.expander("🎵 Music generation details"):
                st.markdown(f"**Prompt used:** {detailed_prompt}")
                
        except Exception as e:
            st.error(f"Error generating music: {str(e)}")
            print(f"Error in music generation: {str(e)}", file=sys.stderr)
            return

# Main UI layout
st.title("🎵 Emotion Music Generator")
st.markdown("""
Enter a sentence describing how you feel, and this app will:
1. Detect your emotion using Llama 3 🤖
2. Generate a piece of **audio** music with Meta's MusicGen 🎶
3. Let you listen to and download your emotional melody 🎧
""")

# Text input area
user_input = st.text_area(
    "💬 What are you feeling?",
    placeholder="e.g., I feel calm and peaceful watching the rain fall...",
    height=150
)

# Place duration slider in the main UI rather than sidebar
duration = st.slider("🕒 Music Duration (seconds)", min_value=5, max_value=30, value=DEFAULT_DURATION, step=5)

# Sidebar for settings
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Advanced emotion settings
    st.subheader("Emotion Processing")
    manual_override = st.checkbox("Manual Emotion Override", value=False)
    st.session_state.manual_override = manual_override
    
    # Only show these if manual override is checked
    if manual_override:
        # Define variables for manual emotion selection
        selected_emotion = st.selectbox("Primary Emotion", all_emotions, index=0)
        secondary_emotion = st.selectbox("Secondary Emotion (optional)", ["None"] + all_emotions, index=0)
        secondary_emotion = None if secondary_emotion == "None" else secondary_emotion
        emotion_intensity = st.slider("Emotional Intensity", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        
        # Store in session state for persistence
        st.session_state.selected_emotion = selected_emotion
        st.session_state.secondary_emotion = secondary_emotion
        st.session_state.emotion_intensity = emotion_intensity
        
        # Add a confirmation button with more emphasis
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirm", type="primary"):
                emotion_result = {
                    "primary": selected_emotion,
                    "secondary": secondary_emotion,
                    "intensity": emotion_intensity,
                    "context": "Manually selected emotions"
                }
                st.session_state.emotion_result = emotion_result
                st.session_state.manual_emotions_applied = True
                # Turn off the regular analysis flag when manual is applied
                st.session_state.emotions_analyzed = False
                st.success("Emotions confirmed!")
        
        # Add a reset button as well
        with col2:
            if st.button("Reset"):
                st.session_state.manual_emotions_applied = False
                st.session_state.emotions_analyzed = False
                st.session_state.emotion_result = None
                st.info("Manual emotions reset")
    else:
        # Reset the flag when manual override is turned off
        st.session_state.manual_emotions_applied = False
    
    # Separate button for loading MusicGen
    st.subheader("Music Generation")
    load_musicgen_button = st.button("Load MusicGen Model")
    musicgen_status = st.empty()
    
    if load_musicgen_button:
        with musicgen_status:
            with st.spinner("Loading MusicGen model... this may take a minute"):
                try:
                    musicgen = load_musicgen_model()
                    st.session_state.musicgen_loaded = True
                    st.session_state.musicgen = musicgen
                    st.success("✅ MusicGen model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error loading MusicGen model: {str(e)}")
                    st.info("Try running the application again. Sometimes PyTorch needs a restart to properly initialize.")
    elif 'musicgen_loaded' in st.session_state:
        musicgen_status.success("✅ MusicGen model is loaded")

# Display manual emotions if they've been confirmed
if st.session_state.get('manual_emotions_applied', False) and 'emotion_result' in st.session_state:
    st.subheader("🎯 Confirmed Emotions:")
    display_emotion_analysis(st.session_state.emotion_result)
    
    # Add a generate music button right after the confirmed emotions
    if st.button("🎵 Generate Music Based on Selected Emotions") and 'musicgen_loaded' in st.session_state:
        generate_and_display_music(
            st.session_state.emotion_result, 
            "Custom emotion selection", # Using a placeholder text since it's manual
            duration
        )
    elif 'musicgen_loaded' not in st.session_state:
        st.info("👈 Please load the MusicGen model from the sidebar first")

# STEP 1: Analyze Emotion Button (only if no manual emotions applied or if we have user input)
elif user_input and not st.session_state.get('manual_emotions_applied', False):
    if st.button("🔍 Analyze Emotion"):
        # Validate API key before making requests
        try:
            # Simple validation request to ensure API key works
            test_headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            test_payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "user", "content": "Test message"}
                ],
                "max_tokens": 10
            }
            
            with st.spinner("Validating API connection..."):
                test_response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=test_headers,
                    json=test_payload
                )
            
            # If we got here without an exception, the API key is working
            
            # Get emotion analysis
            with st.spinner("Analyzing emotions..."):
                emotion_result = detect_complex_emotion_with_api(user_input)
            
            # Store the emotion result in session state for later use
            st.session_state.emotion_result = emotion_result
            st.session_state.analyzed_text = user_input
            st.session_state.emotions_analyzed = True
            
            # Force a rerun to update the UI
            st.rerun()
                    
        except requests.exceptions.RequestException as e:
            st.error(f"API connection failed: {str(e)}")
            st.info("Please check that your API key is correct and that you have an active internet connection.")

# Display the analyzed emotions if they exist
if st.session_state.get('emotions_analyzed', False) and 'emotion_result' in st.session_state and not st.session_state.get('manual_emotions_applied', False):
    st.subheader("🔍 Detected Emotions:")
    display_emotion_analysis(st.session_state.emotion_result)
    
    # Show a message to load MusicGen if not loaded
    if 'musicgen_loaded' not in st.session_state:
        st.info("👈 Now load the MusicGen model from the sidebar to generate music based on this emotion")
    else:
        # Generate music button - SEPARATE FROM THE ANALYSIS BUTTON
        if st.button("🎵 Generate Music Based on Detected Emotions"):
            generate_and_display_music(
                st.session_state.emotion_result, 
                st.session_state.analyzed_text, 
                duration
            )

# Add examples at the bottom
with st.expander("📝 Example Texts to Try (Click to Copy)"):
    st.markdown("**Copy any example below and paste it into the text area above:**")
    
    examples = [
        "I feel a bittersweet nostalgia as I look through old photographs, remembering good times while also feeling the passage of time.",
        "I'm filled with anticipation and excitement about my upcoming journey, though there's a hint of nervousness too.",
        "I feel completely at peace watching the sunset by the ocean, with a gentle breeze and the sound of waves.",
        "I'm experiencing a complex mix of pride and melancholy as my child leaves for college - happy for their future but sad about the change.",
        "I feel conflicted about my career decision - part of me is excited for new opportunities while another part fears leaving my comfort zone.",
        "I'm filled with awe looking at the night sky, feeling small yet connected to something vast and beautiful.",
        "Finally submitted that project at work today. Everyone congratulated me, but I can't help wondering if I could have done better if I'd had more time.",
        "Had coffee with Alex this morning. We talked about the old days and caught up on life. It's been three years since we last met in person.",
        "Mom called to tell me they're selling the house. I've known it was coming, but hearing it made it real."
    ]
    
    # Display each example as text
    for i, example in enumerate(examples):
        st.markdown(f"**Example {i+1}:**")
        st.text(example)
        st.markdown("---")

if __name__ == "__main__":
    # The app is already running if this script is executed directly
    pass