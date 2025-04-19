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
