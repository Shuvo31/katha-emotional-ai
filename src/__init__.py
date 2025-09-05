"""
Katha (কথা) - Emotional AI Voice for Indian Languages

A comprehensive library for generating emotionally expressive AI voices
in Bengali, Hindi, and other Indian languages with cultural awareness.

Author: Shuvojit Das
Email: shuvojitdas2000@gmail.com
"""

__version__ = "0.1.0"
__author__ = "Shuvojit Das"
__email__ = "shuvojitdas2000@gmail.com"
__description__ = "Emotional AI Voice for Indian Languages"

# Import main classes for easy access
try:
    from .katha_core import KathaCore
    from .language_detector import KathaLanguageDetector
    from .emotion_analyzer import KathaEmotionAnalyzer
    from .cultural_mapper import KathaCulturalMappings
    from .tts_engine import KathaEmotionalTTS
    
    __all__ = [
        "KathaCore",
        "KathaLanguageDetector", 
        "KathaEmotionAnalyzer",
        "KathaCulturalMappings",
        "KathaEmotionalTTS"
    ]
    
except ImportError:
    # During development, modules might not be fully implemented yet
    __all__ = []

# Supported languages
SUPPORTED_LANGUAGES = {
    'bn': 'Bengali (বাংলা)',
    'hi': 'Hindi (हिंदी)',
    'en': 'English'
}

# Supported emotions
SUPPORTED_EMOTIONS = [
    'joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral'
]

def get_version():
    """Get the current version of Katha."""
    return __version__

def get_supported_languages():
    """Get list of supported languages."""
    return SUPPORTED_LANGUAGES

def get_supported_emotions():
    """Get list of supported emotions."""
    return SUPPORTED_EMOTIONS

def quick_demo():
    """Quick demonstration of Katha capabilities."""
    print(f"🎭 Katha (কথা) v{__version__}")
    print("Emotional AI Voice for Indian Languages")
    print(f"By {__author__}")
    print()
    print("Supported Languages:")
    for code, name in SUPPORTED_LANGUAGES.items():
        print(f"  {code}: {name}")
    print()
    print("Supported Emotions:")
    for emotion in SUPPORTED_EMOTIONS:
        print(f"  • {emotion.title()}")
    print()
    print("🚀 Get started with:")
    print("  from katha import KathaCore")
    print("  katha = KathaCore()")
    print("  audio, result = katha.process_text('আমি খুব খুশি!')")

if __name__ == "__main__":
    quick_demo()