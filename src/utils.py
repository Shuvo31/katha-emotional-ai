"""
Katha Utilities
Common utility functions for the Katha emotional AI voice system
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import numpy as np

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file or "katha.log")
        ]
    )

# File utilities
def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if it doesn't"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_json(file_path: Union[str, Path]) -> Dict:
    """Load JSON file safely"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"JSON file not found: {file_path}")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON in {file_path}: {e}")
        return {}

def save_json(data: Dict, file_path: Union[str, Path]) -> bool:
    """Save data to JSON file safely"""
    try:
        ensure_directory(Path(file_path).parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.error(f"Failed to save JSON to {file_path}: {e}")
        return False

# Audio utilities
def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level"""
    if len(audio) == 0:
        return audio
    
    # Calculate RMS
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    
    # Convert target dB to linear scale
    target_rms = 10 ** (target_db / 20)
    
    # Normalize
    normalized = audio * (target_rms / rms)
    
    # Prevent clipping
    max_val = np.max(np.abs(normalized))
    if max_val > 1.0:
        normalized = normalized / max_val
    
    return normalized

def apply_fade(audio: np.ndarray, fade_in_ms: int = 50, fade_out_ms: int = 50, 
               sample_rate: int = 22050) -> np.ndarray:
    """Apply fade in/out to audio"""
    if len(audio) == 0:
        return audio
    
    fade_in_samples = int(fade_in_ms * sample_rate / 1000)
    fade_out_samples = int(fade_out_ms * sample_rate / 1000)
    
    # Ensure fade lengths don't exceed audio length
    fade_in_samples = min(fade_in_samples, len(audio) // 2)
    fade_out_samples = min(fade_out_samples, len(audio) // 2)
    
    result = audio.copy()
    
    # Apply fade in
    if fade_in_samples > 0:
        fade_in = np.linspace(0, 1, fade_in_samples)
        result[:fade_in_samples] *= fade_in
    
    # Apply fade out
    if fade_out_samples > 0:
        fade_out = np.linspace(1, 0, fade_out_samples)
        result[-fade_out_samples:] *= fade_out
    
    return result

# Text processing utilities
def clean_text(text: str) -> str:
    """Clean and normalize text for processing"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(text.split())
    
    # Remove special characters that might interfere with TTS
    # Keep punctuation that affects prosody
    import re
    text = re.sub(r'[^\w\s.,!?;:\-\u0980-\u09FF\u0900-\u097F]', '', text)
    
    return text.strip()

def split_sentences(text: str, language: str = 'en') -> List[str]:
    """Split text into sentences based on language"""
    if not text:
        return []
    
    import re
    
    # Language-specific sentence patterns
    patterns = {
        'en': r'[.!?]+\s+',
        'bn': r'[।!?]+\s+',  # Bengali uses । for full stop
        'hi': r'[।!?]+\s+'   # Hindi also uses ।
    }
    
    pattern = patterns.get(language, patterns['en'])
    sentences = re.split(pattern, text)
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

# Language utilities
def detect_script(text: str) -> str:
    """Detect script/writing system of text"""
    if not text:
        return 'unknown'
    
    # Count characters from different scripts
    bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
    devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    
    total_chars = bengali_chars + devanagari_chars + latin_chars
    
    if total_chars == 0:
        return 'unknown'
    
    # Determine dominant script
    if bengali_chars / total_chars > 0.3:
        return 'bengali'
    elif devanagari_chars / total_chars > 0.3:
        return 'devanagari'
    elif latin_chars / total_chars > 0.3:
        return 'latin'
    else:
        return 'mixed'

# Performance utilities
class Timer:
    """Simple timer for performance monitoring"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer"""
        self.end_time = time.time()
        return self
    
    def elapsed(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time or time.time()
        return end - self.start_time
    
    def __enter__(self):
        return self.start()
    
    def __exit__(self, *args):
        self.stop()

# Configuration utilities
class Config:
    """Configuration manager for Katha"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "katha_config.json"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            "language_settings": {
                "default_language": "en",
                "auto_detect": True,
                "fallback_language": "en"
            },
            "emotion_settings": {
                "default_emotion": "neutral",
                "intensity_multiplier": 1.0,
                "confidence_threshold": 0.5
            },
            "tts_settings": {
                "sample_rate": 22050,
                "default_speed": 1.0,
                "normalize_audio": True,
                "apply_fade": True
            },
            "cultural_settings": {
                "respect_formality": True,
                "regional_variations": True,
                "intensity_modifiers": True
            },
            "performance_settings": {
                "cache_models": True,
                "batch_processing": False,
                "max_text_length": 1000
            }
        }
        
        file_config = load_json(self.config_file)
        
        # Merge default and file config
        merged_config = default_config.copy()
        for section, settings in file_config.items():
            if section in merged_config:
                merged_config[section].update(settings)
            else:
                merged_config[section] = settings
        
        return merged_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self) -> bool:
        """Save configuration to file"""
        return save_json(self.config, self.config_file)

# Validation utilities
def validate_language_code(lang_code: str) -> bool:
    """Validate language code"""
    supported_languages = ['bn', 'hi', 'en']
    return lang_code in supported_languages

def validate_emotion(emotion: str) -> bool:
    """Validate emotion name"""
    supported_emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'neutral']
    return emotion.lower() in supported_emotions

def validate_audio_array(audio: np.ndarray) -> bool:
    """Validate audio array"""
    if not isinstance(audio, np.ndarray):
        return False
    
    if len(audio.shape) != 1:
        return False
    
    if audio.dtype not in [np.float32, np.float64]:
        return False
    
    if np.any(np.abs(audio) > 1.1):  # Allow slight clipping tolerance
        return False
    
    return True

# Caching utilities
class SimpleCache:
    """Simple in-memory cache for models and results"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_times = {}
    
    def get(self, key: str) -> Any:
        """Get item from cache"""
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        # Remove oldest items if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_times.clear()

# Error handling
class KathaError(Exception):
    """Base exception for Katha errors"""
    pass

class LanguageDetectionError(KathaError):
    """Error in language detection"""
    pass

class EmotionAnalysisError(KathaError):
    """Error in emotion analysis"""
    pass

class TTSError(KathaError):
    """Error in text-to-speech generation"""
    pass

class CulturalMappingError(KathaError):
    """Error in cultural context mapping"""
    pass

# Testing utilities
def create_test_audio(duration: float = 1.0, sample_rate: int = 22050, 
                     frequency: float = 440.0) -> np.ndarray:
    """Create test audio signal"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = 0.3 * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)

def run_basic_tests():
    """Run basic utility function tests"""
    print("🧪 Running Katha utility tests...")
    
    # Test timer
    with Timer() as timer:
        time.sleep(0.1)
    assert 0.08 < timer.elapsed() < 0.12, "Timer test failed"
    print("✅ Timer test passed")
    
    # Test text cleaning
    dirty_text = "  Hello,   world!  \n\n  "
    clean = clean_text(dirty_text)
    assert clean == "Hello, world!", "Text cleaning test failed"
    print("✅ Text cleaning test passed")
    
    # Test script detection
    bengali_text = "আমি খুব খুশি"
    hindi_text = "मैं बहुत खुश हूं"
    english_text = "I am very happy"
    
    assert detect_script(bengali_text) == "bengali", "Bengali script detection failed"
    assert detect_script(hindi_text) == "devanagari", "Hindi script detection failed"
    assert detect_script(english_text) == "latin", "English script detection failed"
    print("✅ Script detection tests passed")
    
    # Test audio normalization
    test_audio = create_test_audio()
    normalized = normalize_audio(test_audio)
    assert validate_audio_array(normalized), "Audio normalization test failed"
    print("✅ Audio normalization test passed")
    
    print("🎯 All utility tests passed!")

if __name__ == "__main__":
    run_basic_tests()