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

# Function to display emotion analysis results
def display_emotion_analysis(emotion_result):
    """Display the emotion analysis results in the UI"""
    primary_emotion = emotion_result["primary"]
    secondary_emotion = emotion_result["secondary"]
    intensity = emotion_result["intensity"]
    
    # Get descriptions for display
    primary_desc = complex_emotion_map.get(primary_emotion, {}).get("description", primary_emotion.capitalize())
    
    # Display simplified emotional analysis results
    st.markdown(f"### Detected Emotion: {primary_emotion.capitalize()}")
    st.markdown(f"**Description**: {primary_desc}")
    st.markdown(f"**Emotional Intensity**: {intensity:.2f}")
    
    if secondary_emotion:
        secondary_desc = complex_emotion_map.get(secondary_emotion, {}).get("description", secondary_emotion.capitalize())
        st.markdown(f"**Secondary Emotion**: {secondary_emotion.capitalize()} - {secondary_desc}")

# Function to generate and display music - FIXED VERSION
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
        
        try:
            # Generate music with more error handling
            musicgen = st.session_state.musicgen
            
            print(f"Generating music with duration: {duration} seconds...", file=sys.stderr)
            
            # Set generation parameters more safely
            try:
                musicgen.set_generation_params(duration=duration)
            except Exception as e:
                st.warning(f"Could not set generation parameters: {str(e)}. Using defaults.")
                print(f"Error setting generation params: {str(e)}", file=sys.stderr)
            
            # Generate the music
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
                
        except Exception as e:
            st.error(f"Error generating music: {str(e)}")
            print(f"Error in music generation: {str(e)}", file=sys.stderr)
            return
        
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
    
    # Separate button for loading MusicGen - with better error handling
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
                
    except requests.exceptions.RequestException as e:
        st.error(f"API connection failed: {str(e)}")
        st.info("Please check that your API key is correct and that you have an active internet connection.")

# STEP 2: Generate Music Button (only show if emotion has been analyzed)
elif 'emotion_result' in st.session_state:
    # Display the previously analyzed emotion
    st.subheader("🔍 Previously Analyzed Emotion")
    display_emotion_analysis(st.session_state.emotion_result)
    
    # Generate music button
    if st.button("🎵 Generate Music Based on Emotion") and 'musicgen_loaded' in st.session_state:
        generate_and_display_music(
            st.session_state.emotion_result, 
            st.session_state.analyzed_text, 
            duration
        )
    elif 'musicgen_loaded' not in st.session_state:
        st.info("👈 Please load the MusicGen model from the sidebar first")

# Add examples at the bottom with clickable buttons
with st.expander("📝 Example Texts to Try"):
    st.markdown("**Click on an example to view it, then copy and paste it into the text area above:**")
    
    examples = [
        "I feel a bittersweet nostalgia as I look through old photographs, remembering good times while also feeling the passage of time.",
        "I'm filled with anticipation and excitement about my upcoming journey, though there's a hint of nervousness too.",
        "I feel completely at peace watching the sunset by the ocean, with a gentle breeze and the sound of waves.",
        "I'm experiencing a complex mix of pride and melancholy as my child leaves for college - happy for their future but sad about the change.",
        "I feel conflicted about my career decision - part of me is excited for new opportunities while another part fears leaving my comfort zone.",
        "I'm filled with awe looking at the night sky, feeling small yet connected to something vast and beautiful."
    ]
    
    # Initialize state for showing examples if not already present
    if "show_example" not in st.session_state:
        st.session_state.show_example = None
    
    # Create a grid of example buttons
    cols = st.columns(2)
    for i, example in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"Example {i+1}", key=f"example_{i}"):
                st.session_state.show_example = i
    
    # Show the selected example if any
    if st.session_state.show_example is not None:
        example_index = st.session_state.show_example
        st.markdown(f"**Example {example_index+1}:**")
        st.text_area("Copy this text:", value=examples[example_index], height=100, key=f"example_text_{example_index}")
# Add a simple test for the API key at the bottom for debugging
with st.expander("🔧 Debug Information"):
    st.markdown("### API and Model Information")
    st.markdown(f"**LLM Model**: {LLM_MODEL}")
    st.markdown(f"**Music Model**: {MUSIC_MODEL}")
    
    if st.button("Test API Connection"):
        try:
            test_headers = {
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            }
            
            test_payload = {
                "model": LLM_MODEL,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 10
            }
            
            with st.spinner("Testing API connection..."):
                test_response = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=test_headers,
                    json=test_payload
                )
                
                if test_response.status_code == 200:
                    st.success(f"✅ API connection successful! (Status code: {test_response.status_code})")
                    st.json(test_response.json())
                else:
                    st.error(f"❌ API connection failed with status code: {test_response.status_code}")
                    st.text(test_response.text)
                    
        except Exception as e:
            st.error(f"❌ Error testing API connection: {str(e)}")
    
    # Add MusicGen version check
    st.markdown("### MusicGen Version Check")
    if st.button("Check AudioCraft Version"):
        try:
            import pkg_resources
            try:
                ac_version = pkg_resources.get_distribution("audiocraft").version
                st.info(f"AudioCraft version: {ac_version}")
            except pkg_resources.DistributionNotFound:
                st.warning("AudioCraft package not found or version not available")
            
            st.info("Try this command to install the latest compatible version:")
            st.code("pip install audiocraft==1.0.0 --no-deps")
        except Exception as e:
            st.error(f"Error checking package versions: {str(e)}")

if __name__ == "__main__":
    # The app is already running if this script is executed directly
    pass