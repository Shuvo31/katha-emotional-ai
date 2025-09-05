"""
Katha Emotion Analyzer
Multilingual emotion detection with cultural context
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KathaEmotionAnalyzer:
    """Multilingual emotion detection with cultural context"""
    
    def __init__(self):
        # Initialize emotion classifier
        try:
            self.emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            logger.info("Emotion classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load emotion classifier: {e}")
            self.emotion_classifier = None
        
        # Cultural emotion adjustments for authentic expressions
        self.cultural_context = {
            'bn': {
                'joy_words': ['খুশি', 'আনন্দ', 'দারুণ', 'অসাধারণ', 'চমৎকার', 'বাহ', 'সুন্দর'],
                'sadness_words': ['দুঃখ', 'কষ্ট', 'বিষণ্ণ', 'হতাশ', 'মন খারাপ', 'কান্না'],
                'anger_words': ['রাগ', 'ক্রোধ', 'বিরক্ত', 'ঘৃণা', 'রেগে গেছি'],
                'fear_words': ['ভয়', 'আতঙ্ক', 'চিন্তা', 'দুশ্চিন্তা'],
                'surprise_words': ['আশ্চর্য', 'অবাক', 'চমক', 'বিস্ময়', 'হতবাক'],
                'respect_words': ['আপনি', 'দাদা', 'দিদি', 'মা', 'বাবা', 'স্যার', 'ম্যাডাম'],
                'intensity_modifiers': ['খুব', 'অনেক', 'বেশ', 'ভীষণ', 'অত্যন্ত', 'প্রচণ্ড'],
                'cultural_phrases': {
                    'joy': ['মন ভালো', 'খুশিতে আত্মহারা', 'আনন্দে উদ্বেল'],
                    'sadness': ['মন কেমন করছে', 'হৃদয় ভারাক্রান্ত', 'চোখে জল'],
                    'anger': ['রাগে অগ্নিশর্মা', 'চোখ লাল', 'মাথা গরম']
                },
                'intensity_modifier': 1.2  # Bengalis are more expressive
            },
            'hi': {
                'joy_words': ['खुश', 'खुशी', 'आनंद', 'प्रसन्न', 'हर्षित', 'मजा', 'अच्छा'],
                'sadness_words': ['दुःख', 'गम', 'उदास', 'निराश', 'परेशान', 'रोना'],
                'anger_words': ['गुस्सा', 'क्रोध', 'नाराज', 'चिढ़', 'गुस्से में'],
                'fear_words': ['डर', 'भय', 'चिंता', 'घबराहट'],
                'surprise_words': ['आश्चर्य', 'हैरानी', 'अचरज', 'चौंक'],
                'respect_words': ['आप', 'जी', 'साहब', 'जी हाँ', 'श्रीमान', 'श्रीमती'],
                'intensity_modifiers': ['बहुत', 'काफी', 'अत्यधिक', 'बेहद', 'अत्यंत'],
                'cultural_phrases': {
                    'joy': ['दिल खुश', 'मन प्रसन्न', 'खुशी से झूम उठा'],
                    'sadness': ['दिल टूटा', 'मन भारी', 'आंखों में आंसू'],
                    'anger': ['सिर गर्म', 'आग बबूला', 'खून खौल रहा']
                },
                'intensity_modifier': 1.0
            },
            'en': {
                'joy_words': ['happy', 'joyful', 'excited', 'thrilled', 'delighted', 'cheerful'],
                'sadness_words': ['sad', 'depressed', 'upset', 'disappointed', 'heartbroken'],
                'anger_words': ['angry', 'furious', 'mad', 'irritated', 'annoyed'],
                'fear_words': ['scared', 'afraid', 'worried', 'anxious', 'nervous'],
                'surprise_words': ['surprised', 'shocked', 'amazed', 'astonished'],
                'respect_words': ['sir', 'madam', 'please', 'thank you'],
                'intensity_modifiers': ['very', 'extremely', 'really', 'quite', 'absolutely'],
                'cultural_phrases': {},
                'intensity_modifier': 1.0
            }
        }
        
        # Conversation history for context
        self.conversation_history: List[Dict] = []
        self.max_history = 5
        
        # Emotion mapping from model output to our categories
        self.emotion_mapping = {
            'joy': 'joy',
            'happiness': 'joy',
            'love': 'joy',
            'sadness': 'sadness',
            'grief': 'sadness',
            'disappointment': 'sadness',
            'anger': 'anger',
            'annoyance': 'anger',
            'rage': 'anger',
            'fear': 'fear',
            'anxiety': 'fear',
            'worry': 'fear',
            'surprise': 'surprise',
            'amazement': 'surprise',
            'neutral': 'neutral',
            'disgust': 'anger',  # Map disgust to anger for simplicity
        }
    
    def analyze_emotion(self, text: str, language: str) -> Dict:
        """
        Analyze emotion with cultural context
        
        Args:
            text: Input text to analyze
            language: Language code ('bn', 'hi', 'en')
            
        Returns:
            Dict with emotion analysis results
        """
        if not text or not text.strip():
            return self._create_emotion_result('neutral', 0.5, language, 1.0, 'neutral')
        
        # Get base emotion from transformer model
        base_emotion, confidence = self._get_base_emotion(text)
        
        # Apply cultural context analysis
        cultural_emotion = self._apply_cultural_context(text, base_emotion, language)
        
        # Detect respect/formality level
        respect_level = self._detect_respect_level(text, language)
        
        # Calculate emotion intensity
        intensity = self._calculate_intensity(text, cultural_emotion, language)
        
        # Apply conversation context
        final_emotion = self._apply_conversation_context(cultural_emotion, confidence)
        
        emotion_result = self._create_emotion_result(
            final_emotion, confidence, language, intensity, respect_level
        )
        
        # Store in conversation history
        self._update_conversation_history(emotion_result)
        
        return emotion_result
    
    def _get_base_emotion(self, text: str) -> tuple:
        """Get base emotion from transformer model"""
        if not self.emotion_classifier:
            return 'neutral', 0.5
        
        try:
            results = self.emotion_classifier(text)
            if results and len(results) > 0:
                # Get the highest scoring emotion
                best_result = max(results, key=lambda x: x['score'])
                emotion = best_result['label'].lower()
                confidence = best_result['score']
                
                # Map to our emotion categories
                mapped_emotion = self.emotion_mapping.get(emotion, 'neutral')
                return mapped_emotion, confidence
            else:
                return 'neutral', 0.5
                
        except Exception as e:
            logger.error(f"Emotion classification failed: {e}")
            return 'neutral', 0.5
    
    def _apply_cultural_context(self, text: str, base_emotion: str, language: str) -> str:
        """Apply cultural emotion patterns"""
        if language not in self.cultural_context:
            return base_emotion
            
        context = self.cultural_context[language]
        text_lower = text.lower()
        
        # Check for language-specific emotion words (higher priority than base model)
        emotion_word_scores = {}
        for emotion_type in ['joy', 'sadness', 'anger', 'fear', 'surprise']:
            emotion_words = context.get(f'{emotion_type}_words', [])
            score = sum(1 for word in emotion_words if word.lower() in text_lower)
            if score > 0:
                emotion_word_scores[emotion_type] = score
        
        # If we found clear emotion words, use them
        if emotion_word_scores:
            cultural_emotion = max(emotion_word_scores, key=emotion_word_scores.get)
            return cultural_emotion
        
        # Check for cultural phrases
        for emotion_type, phrases in context.get('cultural_phrases', {}).items():
            for phrase in phrases:
                if phrase.lower() in text_lower:
                    return emotion_type
        
        return base_emotion
    
    def _detect_respect_level(self, text: str, language: str) -> str:
        """Detect level of formality/respect"""
        if language not in self.cultural_context:
            return 'neutral'
            
        respect_words = self.cultural_context[language].get('respect_words', [])
        text_lower = text.lower()
        
        respect_count = sum(1 for word in respect_words if word.lower() in text_lower)
        
        if respect_count > 0:
            return 'high'
        elif language == 'hi' and any(word in text_lower for word in ['तुम', 'तू']):
            return 'low'
        elif language == 'bn' and 'তুমি' in text_lower:
            return 'medium'
        elif language == 'bn' and 'তুই' in text_lower:
            return 'low'
        else:
            return 'neutral'
    
    def _calculate_intensity(self, text: str, emotion: str, language: str) -> float:
        """Calculate emotional intensity"""
        base_intensity = 1.0
        
        # Apply language-specific intensity modifier
        if language in self.cultural_context:
            lang_modifier = self.cultural_context[language].get('intensity_modifier', 1.0)
            base_intensity *= lang_modifier
        
        # Check for intensity modifiers
        if language in self.cultural_context:
            intensity_words = self.cultural_context[language].get('intensity_modifiers', [])
            text_lower = text.lower()
            intensity_count = sum(1 for word in intensity_words if word.lower() in text_lower)
            base_intensity *= (1.0 + intensity_count * 0.3)
        
        # Adjust based on punctuation and formatting
        if '!' in text:
            base_intensity *= 1.3
        if text.isupper():
            base_intensity *= 1.5
        if '???' in text or '!!!' in text:
            base_intensity *= 1.4
        if any(char in text for char in ['😊', '😢', '😡', '😱', '😮']):  # Emojis
            base_intensity *= 1.2
            
        return min(base_intensity, 2.5)  # Cap at 2.5
    
    def _apply_conversation_context(self, emotion: str, confidence: float) -> str:
        """Apply conversation context for emotion smoothing"""
        if not self.conversation_history:
            return emotion
        
        # If confidence is low, consider previous emotions
        if confidence < 0.7 and len(self.conversation_history) > 0:
            recent_emotions = [h['emotion'] for h in self.conversation_history[-2:]]
            
            # If recent emotions are consistent, blend with current
            if len(set(recent_emotions)) == 1 and recent_emotions[0] != 'neutral':
                prev_emotion = recent_emotions[0]
                # If current emotion is neutral/weak, lean towards previous
                if emotion == 'neutral' or confidence < 0.5:
                    return prev_emotion
        
        return emotion
    
    def _create_emotion_result(self, emotion: str, confidence: float, 
                             language: str, intensity: float, respect_level: str) -> Dict:
        """Create standardized emotion result"""
        return {
            'emotion': emotion,
            'confidence': float(confidence),
            'language': language,
            'intensity': float(intensity),
            'respect_level': respect_level,
            'timestamp': np.datetime64('now').astype(str)
        }
    
    def _update_conversation_history(self, emotion_result: Dict):
        """Update conversation history"""
        self.conversation_history.append(emotion_result)
        
        # Keep only recent history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_conversation_emotion_trend(self) -> List[str]:
        """Get recent emotion trend"""
        return [h['emotion'] for h in self.conversation_history]
    
    def reset_conversation_history(self):
        """Reset conversation history"""
        self.conversation_history = []
    
    def get_emotion_explanation(self, emotion_result: Dict) -> str:
        """Get human-readable explanation of emotion detection"""
        emotion = emotion_result['emotion']
        language = emotion_result['language']
        confidence = emotion_result['confidence']
        intensity = emotion_result['intensity']
        
        explanations = {
            'bn': {
                'joy': f"আনন্দ প্রকাশ পেয়েছে ({confidence:.0%} নিশ্চিততা)",
                'sadness': f"দুঃখ অনুভূত হয়েছে ({confidence:.0%} নিশ্চিততা)", 
                'anger': f"রাগ প্রকাশ পেয়েছে ({confidence:.0%} নিশ্চিততা)",
                'fear': f"ভয় বা চিন্তা ({confidence:.0%} নিশ্চিততা)",
                'surprise': f"আশ্চর্য প্রকাশ ({confidence:.0%} নিশ্চিততা)",
                'neutral': f"নিরপেক্ষ মনোভাব ({confidence:.0%} নিশ্চিততা)"
            },
            'hi': {
                'joy': f"खुशी व्यक्त की गई ({confidence:.0%} निश्चितता)",
                'sadness': f"दुःख महसूस किया गया ({confidence:.0%} निश्चितता)",
                'anger': f"गुस्सा व्यक्त किया गया ({confidence:.0%} निश्चितता)",
                'fear': f"डर या चिंता ({confidence:.0%} निश्चितता)",
                'surprise': f"आश्चर्य व्यक्त किया ({confidence:.0%} निश्चितता)",
                'neutral': f"तटस्थ भावना ({confidence:.0%} निश्चितता)"
            },
            'en': {
                'joy': f"Joy expressed ({confidence:.0%} confidence)",
                'sadness': f"Sadness detected ({confidence:.0%} confidence)",
                'anger': f"Anger expressed ({confidence:.0%} confidence)",
                'fear': f"Fear or worry ({confidence:.0%} confidence)",
                'surprise': f"Surprise expressed ({confidence:.0%} confidence)",
                'neutral': f"Neutral sentiment ({confidence:.0%} confidence)"
            }
        }
        
        lang_explanations = explanations.get(language, explanations['en'])
        base_explanation = lang_explanations.get(emotion, f"Unknown emotion: {emotion}")
        
        if intensity > 1.5:
            intensity_note = " (High intensity)" if language == 'en' else " (তীব্রতা বেশি)" if language == 'bn' else " (उच्च तीव्रता)"
            base_explanation += intensity_note
            
        return base_explanation


# Test the analyzer
if __name__ == "__main__":
    analyzer = KathaEmotionAnalyzer()
    
    # Test cases
    test_cases = [
        ("আমি খুব খুশি!", "bn"),  # Bengali joy
        ("আমি দুঃখিত", "bn"),  # Bengali sadness
        ("मैं बहुत खुश हूं!", "hi"),  # Hindi joy
        ("मुझे गुस्सा आ रहा है", "hi"),  # Hindi anger
        ("I am very excited!", "en"),  # English joy
        ("আপনি কেমন আছেন দাদা?", "bn"),  # Bengali respect
        ("तुम कैसे हो यार?", "hi"),  # Hindi casual
    ]
    
    for text, lang in test_cases:
        result = analyzer.analyze_emotion(text, lang)
        explanation = analyzer.get_emotion_explanation(result)
        
        print(f"Text: '{text}' ({lang})")
        print(f"Result: {result}")
        print(f"Explanation: {explanation}")
        print("-" * 60)