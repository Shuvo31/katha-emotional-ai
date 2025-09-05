"""
Katha Emotional TTS Engine
Text-to-Speech with emotional prosody control
"""

import numpy as np
import soundfile as sf
import librosa
import torch
from typing import Dict, Optional, Tuple
import tempfile
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KathaEmotionalTTS:
    """Emotional Text-to-Speech with cultural prosody"""
    
    def __init__(self):
        # Initialize TTS models
        self.tts = None
        self.sample_rate = 22050
        
        try:
            # Try to load TTS model
            from TTS.api import TTS
            self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            logger.info("TTS model loaded successfully")
        except Exception as e:
            logger.warning(f"TTS model loading failed: {e}. Using mock TTS for demo.")
            self.tts = None
        
        # Emotion-to-prosody mapping for each language
        self.prosody_configs = {
            'bn': {
                'joy': {'speed': 1.2, 'pitch_shift': 0.15, 'energy': 1.3, 'rhythm': 'melodic'},
                'sadness': {'speed': 0.7, 'pitch_shift': -0.2, 'energy': 0.7, 'rhythm': 'slow'},
                'anger': {'speed': 1.4, 'pitch_shift': 0.1, 'energy': 1.5, 'rhythm': 'sharp'},
                'fear': {'speed': 1.5, 'pitch_shift': 0.25, 'energy': 1.2, 'rhythm': 'trembling'},
                'surprise': {'speed': 1.1, 'pitch_shift': 0.2, 'energy': 1.4, 'rhythm': 'ascending'},
                'neutral': {'speed': 1.0, 'pitch_shift': 0.0, 'energy': 1.0, 'rhythm': 'normal'}
            },
            'hi': {
                'joy': {'speed': 1.1, 'pitch_shift': 0.12, 'energy': 1.2, 'rhythm': 'rhythmic'},
                'sadness': {'speed': 0.8, 'pitch_shift': -0.15, 'energy': 0.8, 'rhythm': 'steady'},
                'anger': {'speed': 1.3, 'pitch_shift': 0.08, 'energy': 1.4, 'rhythm': 'forceful'},
                'fear': {'speed': 1.4, 'pitch_shift': 0.2, 'energy': 1.1, 'rhythm': 'nervous'},
                'surprise': {'speed': 1.05, 'pitch_shift': 0.18, 'energy': 1.3, 'rhythm': 'rising'},
                'neutral': {'speed': 1.0, 'pitch_shift': 0.0, 'energy': 1.0, 'rhythm': 'normal'}
            },
            'en': {
                'joy': {'speed': 1.15, 'pitch_shift': 0.1, 'energy': 1.25, 'rhythm': 'bouncy'},
                'sadness': {'speed': 0.75, 'pitch_shift': -0.12, 'energy': 0.75, 'rhythm': 'dragging'},
                'anger': {'speed': 1.25, 'pitch_shift': 0.05, 'energy': 1.35, 'rhythm': 'harsh'},
                'fear': {'speed': 1.35, 'pitch_shift': 0.15, 'energy': 1.1, 'rhythm': 'shaky'},
                'surprise': {'speed': 1.0, 'pitch_shift': 0.15, 'energy': 1.2, 'rhythm': 'jumping'},
                'neutral': {'speed': 1.0, 'pitch_shift': 0.0, 'energy': 1.0, 'rhythm': 'normal'}
            }
        }
        
        # Voice characteristics for different languages
        self.voice_characteristics = {
            'bn': {
                'base_pitch': 180,  # Higher for Bengali melodic nature
                'pitch_variability': 0.3,
                'rhythm_patterns': ['melodic', 'flowing', 'expressive']
            },
            'hi': {
                'base_pitch': 160,  # Standard pitch for Hindi
                'pitch_variability': 0.25,
                'rhythm_patterns': ['rhythmic', 'measured', 'clear']
            },
            'en': {
                'base_pitch': 150,  # Lower base pitch for English
                'pitch_variability': 0.2,
                'rhythm_patterns': ['steady', 'direct', 'clear']
            }
        }
    
    def synthesize_emotional_speech(self, text: str, emotion_data: Dict) -> np.ndarray:
        """
        Generate emotional speech from text
        
        Args:
            text: Input text to synthesize
            emotion_data: Emotion analysis result containing emotion, language, intensity, etc.
            
        Returns:
            numpy array of audio samples
        """
        if not text or not text.strip():
            return np.array([])
        
        emotion = emotion_data.get('emotion', 'neutral')
        language = emotion_data.get('language', 'en')
        intensity = emotion_data.get('intensity', 1.0)
        respect_level = emotion_data.get('respect_level', 'neutral')
        
        try:
            # Generate base audio
            if self.tts:
                wav = self._generate_base_audio(text)
            else:
                # Generate mock audio for demo
                wav = self._generate_mock_audio(text, emotion)
            
            if len(wav) == 0:
                return np.array([])
            
            # Apply emotional modifications
            modified_wav = self._apply_emotional_prosody(
                wav, emotion, language, intensity, respect_level
            )
            
            return modified_wav
            
        except Exception as e:
            logger.error(f"TTS generation failed: {str(e)}")
            return np.array([])
    
    def _generate_base_audio(self, text: str) -> np.ndarray:
        """Generate base audio using TTS model"""
        try:
            wav = self.tts.tts(text)
            return np.array(wav)
        except Exception as e:
            logger.error(f"Base audio generation failed: {e}")
            return self._generate_mock_audio(text, 'neutral')
    
    def _generate_mock_audio(self, text: str, emotion: str) -> np.ndarray:
        """Generate mock audio for demonstration purposes"""
        # Create a simple sine wave based on text length and emotion
        duration = max(0.5, len(text) * 0.1)  # Roughly 0.1 seconds per character
        
        # Base frequency varies by emotion
        emotion_frequencies = {
            'joy': 440,      # A4 - bright
            'sadness': 220,  # A3 - lower
            'anger': 330,    # E4 - sharp
            'fear': 500,     # slightly higher, nervous
            'surprise': 550, # high pitch
            'neutral': 330   # middle
        }
        
        base_freq = emotion_frequencies.get(emotion, 330)
        t = np.linspace(0, duration, int(self.sample_rate * duration))
        
        # Create a more complex waveform (not just sine)
        wave = (np.sin(2 * np.pi * base_freq * t) + 
                0.3 * np.sin(2 * np.pi * base_freq * 2 * t) +
                0.1 * np.sin(2 * np.pi * base_freq * 3 * t))
        
        # Add some variation to make it more speech-like
        modulation = np.sin(2 * np.pi * 5 * t) * 0.1  # 5Hz modulation
        wave = wave * (1 + modulation)
        
        # Apply envelope (fade in/out)
        envelope = np.ones_like(wave)
        fade_samples = int(0.05 * self.sample_rate)  # 50ms fade
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        wave = wave * envelope * 0.3  # Reduce volume
        return wave.astype(np.float32)
    
    def _apply_emotional_prosody(self, wav: np.ndarray, emotion: str, 
                                language: str, intensity: float, respect_level: str) -> np.ndarray:
        """Apply emotional modifications to audio"""
        
        if language not in self.prosody_configs or emotion not in self.prosody_configs[language]:
            return wav
        
        config = self.prosody_configs[language][emotion].copy()
        
        # Adjust based on intensity
        config['speed'] = 1.0 + (config['speed'] - 1.0) * min(intensity, 1.5)
        config['pitch_shift'] *= min(intensity, 1.5)
        config['energy'] = 1.0 + (config['energy'] - 1.0) * min(intensity, 1.5)
        
        # Adjust for respect level
        if respect_level == 'high':
            config['speed'] *= 0.9  # Slower for respect
            config['energy'] *= 0.9  # Softer
        elif respect_level == 'low':
            config['speed'] *= 1.1  # Faster for casual
            config['energy'] *= 1.05  # Slightly more energetic
        
        # Apply modifications
        modified_wav = wav.copy()
        
        # Speed modification
        if abs(config['speed'] - 1.0) > 0.05:
            modified_wav = self._change_speed(modified_wav, config['speed'])
        
        # Pitch modification
        if abs(config['pitch_shift']) > 0.01:
            modified_wav = self._pitch_shift(modified_wav, config['pitch_shift'])
        
        # Energy modification
        if abs(config['energy'] - 1.0) > 0.05:
            modified_wav = modified_wav * config['energy']
            modified_wav = np.clip(modified_wav, -1.0, 1.0)
        
        # Apply rhythm patterns
        modified_wav = self._apply_rhythm_pattern(modified_wav, config.get('rhythm', 'normal'))
        
        return modified_wav
    
    def _change_speed(self, wav: np.ndarray, speed_factor: float) -> np.ndarray:
        """Change playback speed using librosa"""
        try:
            return librosa.effects.time_stretch(wav, rate=speed_factor)
        except Exception as e:
            logger.warning(f"Speed change failed: {e}")
            return wav
    
    def _pitch_shift(self, wav: np.ndarray, semitones: float) -> np.ndarray:
        """Shift pitch by semitones using librosa"""
        try:
            return librosa.effects.pitch_shift(wav, sr=self.sample_rate, n_steps=semitones)
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return wav
    
    def _apply_rhythm_pattern(self, wav: np.ndarray, rhythm: str) -> np.ndarray:
        """Apply rhythm-specific modifications"""
        if rhythm == 'melodic':
            # Add slight vibrato for melodic speech
            t = np.linspace(0, len(wav)/self.sample_rate, len(wav))
            vibrato = 1 + 0.02 * np.sin(2 * np.pi * 6 * t)  # 6Hz vibrato
            return wav * vibrato
        
        elif rhythm == 'sharp':
            # Add slight compression for sharper sound
            compressed = np.sign(wav) * (np.abs(wav) ** 0.7)
            return compressed
        
        elif rhythm == 'trembling':
            # Add tremolo effect for fear
            t = np.linspace(0, len(wav)/self.sample_rate, len(wav))
            tremolo = 1 + 0.1 * np.sin(2 * np.pi * 8 * t)  # 8Hz tremolo
            return wav * tremolo
        
        elif rhythm == 'ascending':
            # Gradual pitch rise for surprise
            t = np.linspace(0, len(wav)/self.sample_rate, len(wav))
            pitch_envelope = 1 + 0.05 * t / t[-1]  # Gradual rise
            # This is a simplified version - real implementation would need FFT
            return wav * pitch_envelope
        
        else:
            return wav  # No modification for normal/other rhythms
    
    def save_audio(self, audio: np.ndarray, filename: str = None) -> str:
        """
        Save audio to file
        
        Args:
            audio: Audio samples
            filename: Output filename (optional, will create temp file if None)
            
        Returns:
            Path to saved audio file
        """
        if filename is None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            filename = temp_file.name
            temp_file.close()
        
        try:
            if len(audio) > 0:
                sf.write(filename, audio, self.sample_rate)
            else:
                # Create silent file if no audio
                silent_audio = np.zeros(int(0.5 * self.sample_rate))
                sf.write(filename, silent_audio, self.sample_rate)
            
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            return ""
    
    def get_audio_info(self, audio: np.ndarray) -> Dict:
        """Get information about audio"""
        if len(audio) == 0:
            return {'duration': 0, 'sample_rate': self.sample_rate, 'channels': 0}
        
        return {
            'duration': len(audio) / self.sample_rate,
            'sample_rate': self.sample_rate,
            'channels': 1,
            'max_amplitude': float(np.max(np.abs(audio))),
            'rms_energy': float(np.sqrt(np.mean(audio ** 2)))
        }
    
    def synthesize_with_ssml(self, ssml_text: str, emotion_data: Dict) -> np.ndarray:
        """
        Synthesize speech with SSML markup for advanced prosody control
        This is a simplified implementation - full SSML would require more complex parsing
        """
        # For now, just extract text from basic SSML and apply emotion
        import re
        
        # Remove basic SSML tags and extract text
        text = re.sub(r'<[^>]+>', '', ssml_text)
        
        # Look for prosody hints in SSML
        rate_match = re.search(r'rate="([^"]*)"', ssml_text)
        pitch_match = re.search(r'pitch="([^"]*)"', ssml_text)
        
        # Apply SSML modifications to emotion data
        modified_emotion_data = emotion_data.copy()
        
        if rate_match:
            rate_value = rate_match.group(1)
            if 'fast' in rate_value:
                modified_emotion_data['intensity'] = modified_emotion_data.get('intensity', 1.0) * 1.2
            elif 'slow' in rate_value:
                modified_emotion_data['intensity'] = modified_emotion_data.get('intensity', 1.0) * 0.8
        
        return self.synthesize_emotional_speech(text, modified_emotion_data)
    
    def create_emotion_sample(self, text: str, emotions: list, language: str) -> Dict[str, np.ndarray]:
        """
        Create audio samples for multiple emotions with the same text
        Useful for comparison and demonstration
        """
        samples = {}
        
        for emotion in emotions:
            emotion_data = {
                'emotion': emotion,
                'language': language,
                'intensity': 1.2,
                'respect_level': 'neutral'
            }
            
            audio = self.synthesize_emotional_speech(text, emotion_data)
            samples[emotion] = audio
        
        return samples
    
    def blend_emotions(self, text: str, emotion1: str, emotion2: str, 
                      blend_ratio: float, language: str) -> np.ndarray:
        """
        Create audio that blends two emotions
        blend_ratio: 0.0 = all emotion1, 1.0 = all emotion2
        """
        # Generate audio for both emotions
        emotion_data1 = {'emotion': emotion1, 'language': language, 'intensity': 1.0, 'respect_level': 'neutral'}
        emotion_data2 = {'emotion': emotion2, 'language': language, 'intensity': 1.0, 'respect_level': 'neutral'}
        
        audio1 = self.synthesize_emotional_speech(text, emotion_data1)
        audio2 = self.synthesize_emotional_speech(text, emotion_data2)
        
        # Ensure same length
        min_length = min(len(audio1), len(audio2))
        if min_length == 0:
            return np.array([])
        
        audio1 = audio1[:min_length]
        audio2 = audio2[:min_length]
        
        # Blend
        blended = audio1 * (1 - blend_ratio) + audio2 * blend_ratio
        return blended


# Test the TTS engine
if __name__ == "__main__":
    tts_engine = KathaEmotionalTTS()
    
    # Test cases
    test_cases = [
        {
            'text': "আমি খুব খুশি!",
            'emotion_data': {
                'emotion': 'joy',
                'language': 'bn',
                'intensity': 1.5,
                'respect_level': 'neutral'
            }
        },
        {
            'text': "मैं बहुत दुःखी हूं",
            'emotion_data': {
                'emotion': 'sadness',
                'language': 'hi',
                'intensity': 1.2,
                'respect_level': 'high'
            }
        },
        {
            'text': "I am very surprised!",
            'emotion_data': {
                'emotion': 'surprise',
                'language': 'en',
                'intensity': 1.8,
                'respect_level': 'neutral'
            }
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\nTest Case {i+1}: {case['text']}")
        
        # Generate emotional speech
        audio = tts_engine.synthesize_emotional_speech(case['text'], case['emotion_data'])
        
        # Get audio info
        info = tts_engine.get_audio_info(audio)
        print(f"Audio Info: {info}")
        
        # Save audio
        filename = f"test_audio_{i+1}_{case['emotion_data']['emotion']}.wav"
        saved_path = tts_engine.save_audio(audio, filename)
        print(f"Audio saved to: {saved_path}")
    
    # Test emotion comparison
    print("\nGenerating emotion comparison samples...")
    emotions = ['joy', 'sadness', 'anger', 'surprise', 'neutral']
    samples = tts_engine.create_emotion_sample("Hello, how are you?", emotions, 'en')
    
    for emotion, audio in samples.items():
        filename = f"emotion_sample_{emotion}.wav"
        tts_engine.save_audio(audio, filename)
        print(f"Saved {emotion} sample: {filename}")
    
    print("\nKatha TTS Engine test completed!")