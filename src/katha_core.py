"""
Katha Core Engine
Main orchestrator for emotional AI voice processing
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
import time

from .language_detector import KathaLanguageDetector
from .emotion_analyzer import KathaEmotionAnalyzer
from .cultural_mapper import KathaCulturalMappings
from .tts_engine import KathaEmotionalTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KathaCore:
    """
    Main Katha engine combining all components for emotional voice AI
    """
    
    def __init__(self):
        """Initialize all Katha components"""
        logger.info("Initializing Katha Core Engine...")
        
        # Initialize components
        try:
            self.language_detector = KathaLanguageDetector()
            logger.info("✓ Language detector loaded")
            
            self.emotion_analyzer = KathaEmotionAnalyzer()
            logger.info("✓ Emotion analyzer loaded")
            
            self.cultural_mapper = KathaCulturalMappings()
            logger.info("✓ Cultural mapper loaded")
            
            self.tts_engine = KathaEmotionalTTS()
            logger.info("✓ TTS engine loaded")
            
            logger.info("🎭 Katha Core Engine initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize Katha Core: {e}")
            raise
        
        # Performance tracking
        self.processing_stats = {
            'total_requests': 0,
            'avg_processing_time': 0.0,
            'language_distribution': {'bn': 0, 'hi': 0, 'en': 0},
            'emotion_distribution': {
                'joy': 0, 'sadness': 0, 'anger': 0, 
                'fear': 0, 'surprise': 0, 'neutral': 0
            }
        }
    
    def process_text(self, text: str, user_context: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Main processing pipeline: text → emotional voice audio
        
        Args:
            text: Input text to process
            user_context: Optional user context (history, preferences, etc.)
            
        Returns:
            Tuple of (audio_array, detailed_analysis)
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return np.array([]), self._create_empty_result()
        
        try:
            # Step 1: Language Detection
            language = self.language_detector.detect_language(text)
            logger.debug(f"Detected language: {language}")
            
            # Step 2: Emotion Analysis
            emotion_data = self.emotion_analyzer.analyze_emotion(text, language)
            logger.debug(f"Emotion analysis: {emotion_data}")
            
            # Step 3: Cultural Context Analysis
            cultural_context = self.cultural_mapper.detect_cultural_context(text, language)
            logger.debug(f"Cultural context: {cultural_context}")
            
            # Step 4: Enhance emotion data with cultural context
            enhanced_emotion_data = self._enhance_emotion_with_culture(
                emotion_data, cultural_context, user_context
            )
            
            # Step 5: Generate Emotional Speech
            audio = self.tts_engine.synthesize_emotional_speech(text, enhanced_emotion_data)
            
            # Step 6: Create comprehensive result
            result = self._create_comprehensive_result(
                text, audio, enhanced_emotion_data, cultural_context, language
            )
            
            # Update statistics
            self._update_processing_stats(language, enhanced_emotion_data['emotion'], time.time() - start_time)
            
            return audio, result
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return np.array([]), self._create_error_result(str(e))
    
    def process_conversation(self, conversation: list) -> list:
        """
        Process a full conversation with emotion continuity
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            List of processed results with audio
        """
        results = []
        
        for turn in conversation:
            text = turn.get('text', '')
            speaker = turn.get('speaker', 'user')
            
            # Maintain conversation context
            user_context = {
                'conversation_history': results[-3:] if results else [],
                'speaker': speaker
            }
            
            audio, result = self.process_text(text, user_context)
            
            result['speaker'] = speaker
            result['audio'] = audio
            results.append(result)
        
        return results
    
    def generate_response(self, user_text: str, response_emotion: Optional[str] = None) -> Tuple[str, np.ndarray, Dict]:
        """
        Generate contextually appropriate response to user input
        
        Args:
            user_text: User's input text
            response_emotion: Desired response emotion (optional)
            
        Returns:
            Tuple of (response_text, audio, analysis)
        """
        # Analyze user input
        user_language = self.language_detector.detect_language(user_text)
        user_emotion_data = self.emotion_analyzer.analyze_emotion(user_text, user_language)
        cultural_context = self.cultural_mapper.detect_cultural_context(user_text, user_language)
        
        # Generate appropriate response
        if response_emotion is None:
            response_emotion = self._determine_response_emotion(user_emotion_data['emotion'])
        
        response_text = self.cultural_mapper.get_cultural_response(
            user_emotion_data['emotion'], user_language, cultural_context
        )
        
        # Generate response audio
        response_emotion_data = {
            'emotion': response_emotion,
            'language': user_language,
            'intensity': min(user_emotion_data['intensity'] * 0.8, 1.5),  # Slightly calmer response
            'respect_level': cultural_context.get('formality', 'neutral')
        }
        
        response_audio, response_analysis = self.process_text(response_text)
        
        return response_text, response_audio, response_analysis
    
    def analyze_text_only(self, text: str) -> Dict:
        """
        Analyze text without generating audio (faster for analysis-only use cases)
        """
        language = self.language_detector.detect_language(text)
        emotion_data = self.emotion_analyzer.analyze_emotion(text, language)
        cultural_context = self.cultural_mapper.detect_cultural_context(text, language)
        
        return {
            'text': text,
            'language': language,
            'emotion_data': emotion_data,
            'cultural_context': cultural_context,
            'language_confidence': self.language_detector.detect_with_confidence(text),
            'is_multilingual': self.language_detector.is_multilingual(text)
        }
    
    def create_voice_samples(self, text: str, emotions: list = None, languages: list = None) -> Dict:
        """
        Create multiple voice samples for comparison/demonstration
        
        Args:
            text: Text to synthesize
            emotions: List of emotions to generate (default: all)
            languages: List of languages to generate (default: all supported)
            
        Returns:
            Dict with samples organized by language and emotion
        """
        if emotions is None:
            emotions = ['joy', 'sadness', 'anger', 'surprise', 'neutral']
        
        if languages is None:
            languages = ['bn', 'hi', 'en']
        
        samples = {}
        
        for language in languages:
            samples[language] = {}
            for emotion in emotions:
                emotion_data = {
                    'emotion': emotion,
                    'language': language,
                    'intensity': 1.2,
                    'respect_level': 'neutral'
                }
                
                audio = self.tts_engine.synthesize_emotional_speech(text, emotion_data)
                samples[language][emotion] = {
                    'audio': audio,
                    'info': self.tts_engine.get_audio_info(audio)
                }
        
        return samples
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return self.processing_stats.copy()
    
    def reset_conversation_history(self):
        """Reset conversation history in emotion analyzer"""
        self.emotion_analyzer.reset_conversation_history()
    
    def _enhance_emotion_with_culture(self, emotion_data: Dict, cultural_context: Dict, 
                                    user_context: Optional[Dict] = None) -> Dict:
        """Enhance emotion data with cultural and user context"""
        enhanced = emotion_data.copy()
        
        # Apply cultural intensity modifications
        cultural_intensity = cultural_context.get('emotion_intensity', 1.0)
        enhanced['intensity'] *= cultural_intensity
        
        # Apply formality adjustments
        if cultural_context.get('formality') == 'high_respect':
            enhanced['respect_level'] = 'high'
            enhanced['intensity'] *= 0.9  # More subdued for formal contexts
        elif cultural_context.get('formality') == 'casual':
            enhanced['respect_level'] = 'low'
            enhanced['intensity'] *= 1.1  # More expressive for casual
        
        # Add cultural markers
        enhanced['cultural_context'] = cultural_context
        
        # Apply user context if provided
        if user_context:
            conversation_history = user_context.get('conversation_history', [])
            if conversation_history:
                # Adjust based on conversation flow
                recent_emotions = [h.get('emotion_data', {}).get('emotion', 'neutral') 
                                 for h in conversation_history[-2:]]
                
                # If conversation was consistently sad, moderate the intensity
                if recent_emotions.count('sadness') >= 1 and enhanced['emotion'] != 'joy':
                    enhanced['intensity'] *= 0.8
        
        return enhanced
    
    def _determine_response_emotion(self, user_emotion: str) -> str:
        """Determine appropriate response emotion based on user emotion"""
        response_mapping = {
            'joy': 'joy',        # Match joy with joy
            'sadness': 'neutral', # Respond to sadness with calm support
            'anger': 'neutral',   # Respond to anger with calm
            'fear': 'neutral',    # Respond to fear with reassurance
            'surprise': 'joy',    # Respond to surprise with positive engagement
            'neutral': 'neutral'  # Match neutral
        }
        
        return response_mapping.get(user_emotion, 'neutral')
    
    def _create_comprehensive_result(self, text: str, audio: np.ndarray, 
                                   emotion_data: Dict, cultural_context: Dict, language: str) -> Dict:
        """Create comprehensive analysis result"""
        return {
            'text': text,
            'language': language,
            'language_name': self.language_detector.get_language_name(language),
            'emotion_data': emotion_data,
            'cultural_context': cultural_context,
            'audio_info': self.tts_engine.get_audio_info(audio),
            'processing_time': time.time(),
            'confidence_scores': self.language_detector.detect_with_confidence(text),
            'emotion_explanation': self.emotion_analyzer.get_emotion_explanation(emotion_data),
            'cultural_markers': cultural_context.get('cultural_markers', []),
            'prosody_config': self.cultural_mapper.get_prosody_config(
                emotion_data['emotion'], language, cultural_context
            )
        }
    
    def _create_empty_result(self) -> Dict:
        """Create result for empty input"""
        return {
            'text': '',
            'language': 'en',
            'emotion_data': {'emotion': 'neutral', 'confidence': 0.0},
            'cultural_context': {},
            'audio_info': {'duration': 0},
            'error': 'Empty input'
        }
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create result for processing errors"""
        return {
            'text': '',
            'language': 'en',
            'emotion_data': {'emotion': 'neutral', 'confidence': 0.0},
            'cultural_context': {},
            'audio_info': {'duration': 0},
            'error': error_message
        }
    
    def _update_processing_stats(self, language: str, emotion: str, processing_time: float):
        """Update processing statistics"""
        self.processing_stats['total_requests'] += 1
        
        # Update average processing time
        total = self.processing_stats['total_requests']
        current_avg = self.processing_stats['avg_processing_time']
        self.processing_stats['avg_processing_time'] = (current_avg * (total - 1) + processing_time) / total
        
        # Update language distribution
        if language in self.processing_stats['language_distribution']:
            self.processing_stats['language_distribution'][language] += 1
        
        # Update emotion distribution
        if emotion in self.processing_stats['emotion_distribution']:
            self.processing_stats['emotion_distribution'][emotion] += 1


# Convenience function for quick usage
def quick_process(text: str) -> Tuple[np.ndarray, str]:
    """
    Quick processing function for simple use cases
    
    Returns:
        Tuple of (audio, emotion)
    """
    katha = KathaCore()
    audio, result = katha.process_text(text)
    emotion = result.get('emotion_data', {}).get('emotion', 'neutral')
    return audio, emotion


# Test the core engine
if __name__ == "__main__":
    # Initialize Katha
    katha = KathaCore()
    
    # Test cases
    test_texts = [
        "আমি খুব খুশি!",                    # Bengali joy
        "मैं बहुत परेशान हूं",              # Hindi sadness  
        "I am absolutely thrilled!",        # English joy
        "আপনি কেমন আছেন দাদা?",            # Bengali respect
        "तुम कैसे हो यार?",                # Hindi casual
    ]
    
    print("🎭 Testing Katha Core Engine\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: '{text}'")
        
        # Process text
        audio, result = katha.process_text(text)
        
        # Display results
        print(f"  Language: {result.get('language_name', 'Unknown')}")
        print(f"  Emotion: {result.get('emotion_data', {}).get('emotion', 'unknown').title()}")
        print(f"  Confidence: {result.get('emotion_data', {}).get('confidence', 0):.0%}")
        print(f"  Cultural Context: {result.get('cultural_context', {}).get('formality', 'unknown')}")
        print(f"  Audio Duration: {result.get('audio_info', {}).get('duration', 0):.2f}s")
        print(f"  Explanation: {result.get('emotion_explanation', 'N/A')}")
        
        # Save audio
        filename = f"katha_test_{i}.wav"
        katha.tts_engine.save_audio(audio, filename)
        print(f"  Audio saved: {filename}")
        print("-" * 60)
    
    # Test conversation processing
    print("\n🗣️ Testing Conversation Processing\n")
    
    conversation = [
        {'text': 'আমি আজ খুব দুঃখিত', 'speaker': 'user'},
        {'text': 'কি হয়েছে? বলুন না।', 'speaker': 'assistant'},
        {'text': 'আমার চাকরি চলে গেছে', 'speaker': 'user'},
        {'text': 'খুব দুঃখ লাগছে। কিন্তু নতুন সুযোগ আসবে।', 'speaker': 'assistant'}
    ]
    
    conversation_results = katha.process_conversation(conversation)
    
    for i, result in enumerate(conversation_results):
        speaker = result.get('speaker', 'unknown')
        emotion = result.get('emotion_data', {}).get('emotion', 'unknown')
        print(f"Turn {i+1} ({speaker}): {emotion.title()} emotion")
    
    # Display statistics
    print(f"\n📊 Processing Statistics:")
    stats = katha.get_processing_stats()
    print(f"  Total Requests: {stats['total_requests']}")
    print(f"  Avg Processing Time: {stats['avg_processing_time']:.3f}s")
    print(f"  Language Distribution: {stats['language_distribution']}")
    print(f"  Emotion Distribution: {stats['emotion_distribution']}")
    
    print("\n🎯 Katha Core Engine test completed!")