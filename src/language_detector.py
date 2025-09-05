"""
Katha Language Detector
Multi-language detection for Bengali, Hindi, and English
"""

import re
from typing import Dict, List


class KathaLanguageDetector:
    """Detects Bengali, Hindi, and English with high accuracy"""
    
    def __init__(self):
        self.language_patterns = {
            'bn': r'[\u0980-\u09FF]',  # Bengali Unicode range
            'hi': r'[\u0900-\u097F]',  # Devanagari (Hindi) Unicode range
            'en': r'[a-zA-Z]'
        }
        
        # Common words for better detection accuracy
        self.language_words = {
            'bn': [
                'আমি', 'তুমি', 'আপনি', 'কি', 'কেমন', 'আছেন', 'খুব', 'ভালো', 
                'খুশি', 'দুঃখ', 'নমস্কার', 'ধন্যবাদ', 'দাদা', 'দিদি', 'মা', 'বাবা',
                'এটা', 'ওটা', 'কোথায়', 'কখন', 'কেন', 'কিভাবে', 'আজ', 'কাল'
            ],
            'hi': [
                'मैं', 'तुम', 'आप', 'क्या', 'कैसे', 'हैं', 'बहुत', 'अच्छा', 
                'खुश', 'दुःख', 'नमस्ते', 'धन्यवाद', 'भाई', 'जी', 'साहब',
                'यह', 'वह', 'कहाँ', 'कब', 'क्यों', 'कैसे', 'आज', 'कल'
            ],
            'en': [
                'i', 'you', 'how', 'are', 'very', 'good', 'happy', 'sad',
                'hello', 'thanks', 'today', 'tomorrow', 'what', 'where', 'when'
            ]
        }
        
        # Language-specific punctuation and symbols
        self.language_symbols = {
            'bn': ['।', '॥'],  # Bengali punctuation
            'hi': ['।', '॥'],  # Devanagari punctuation
            'en': ['.', '!', '?', ',']
        }
    
    def detect_language(self, text: str) -> str:
        """
        Detect primary language in text
        Returns: 'bn' for Bengali, 'hi' for Hindi, 'en' for English
        """
        if not text or not text.strip():
            return 'en'  # Default to English for empty text
            
        text_lower = text.lower().strip()
        
        # Count characters for each language
        char_counts = {}
        for lang, pattern in self.language_patterns.items():
            char_counts[lang] = len(re.findall(pattern, text))
        
        # Boost score for common words (more reliable than just character counting)
        word_scores = {lang: 0 for lang in self.language_words}
        for lang, words in self.language_words.items():
            for word in words:
                if word.lower() in text_lower:
                    word_scores[lang] += 3  # Higher weight for word matches
        
        # Check for language-specific symbols
        symbol_scores = {lang: 0 for lang in self.language_symbols}
        for lang, symbols in self.language_symbols.items():
            for symbol in symbols:
                if symbol in text:
                    symbol_scores[lang] += 1
        
        # Combine all scores
        total_scores = {}
        for lang in char_counts:
            total_scores[lang] = (
                char_counts[lang] + 
                word_scores[lang] * 2 +  # Word matches are more important
                symbol_scores[lang]
            )
        
        # Special case: if no clear indicators, check for common patterns
        if all(score == 0 for score in total_scores.values()):
            return self._fallback_detection(text)
        
        # Return language with highest score
        detected_lang = max(total_scores, key=total_scores.get)
        
        # Confidence check: if score is very low, default to English
        if total_scores[detected_lang] < 2:
            return 'en'
            
        return detected_lang
    
    def _fallback_detection(self, text: str) -> str:
        """Fallback detection for edge cases"""
        # Check for English-like patterns
        if re.search(r'\b(the|and|or|but|in|on|at|to|for|of|with)\b', text.lower()):
            return 'en'
        
        # Check for romanized Indian language patterns
        if re.search(r'(aap|tum|mai|ki|ke|ko|se|me)', text.lower()):
            return 'hi'  # Likely romanized Hindi
            
        return 'en'  # Default fallback
    
    def detect_with_confidence(self, text: str) -> Dict[str, float]:
        """
        Detect language with confidence scores for all languages
        Returns: Dict with language codes and confidence scores
        """
        if not text or not text.strip():
            return {'bn': 0.0, 'hi': 0.0, 'en': 1.0}
        
        text_lower = text.lower().strip()
        scores = {'bn': 0.0, 'hi': 0.0, 'en': 0.0}
        
        # Character-based scoring
        for lang, pattern in self.language_patterns.items():
            char_count = len(re.findall(pattern, text))
            scores[lang] += char_count * 0.5
        
        # Word-based scoring (more reliable)
        for lang, words in self.language_words.items():
            for word in words:
                if word.lower() in text_lower:
                    scores[lang] += 2.0
        
        # Normalize scores to percentages
        total_score = sum(scores.values())
        if total_score > 0:
            for lang in scores:
                scores[lang] = scores[lang] / total_score
        else:
            scores['en'] = 1.0  # Default to English if no matches
        
        return scores
    
    def is_multilingual(self, text: str, threshold: float = 0.3) -> bool:
        """
        Check if text contains multiple languages
        Returns: True if multiple languages detected above threshold
        """
        confidence_scores = self.detect_with_confidence(text)
        languages_above_threshold = sum(
            1 for score in confidence_scores.values() 
            if score >= threshold
        )
        return languages_above_threshold > 1
    
    def get_language_name(self, lang_code: str) -> str:
        """Get full language name from code"""
        lang_names = {
            'bn': 'Bengali (বাংলা)',
            'hi': 'Hindi (हिंदी)', 
            'en': 'English'
        }
        return lang_names.get(lang_code, 'Unknown')


# Test the detector
if __name__ == "__main__":
    detector = KathaLanguageDetector()
    
    # Test cases
    test_texts = [
        "আমি খুব খুশি!",  # Bengali
        "मैं बहुत खुश हूं!",  # Hindi
        "I am very happy!",  # English
        "আপনি কেমন আছেন?",  # Bengali formal
        "तुम कैसे हो?",  # Hindi casual
        "Hello, কেমন আছেন? How are you?",  # Mixed
        "",  # Empty
        "123 !@#",  # No text
    ]
    
    for text in test_texts:
        lang = detector.detect_language(text)
        confidence = detector.detect_with_confidence(text)
        multilingual = detector.is_multilingual(text)
        
        print(f"Text: '{text}'")
        print(f"Detected: {lang} ({detector.get_language_name(lang)})")
        print(f"Confidence: {confidence}")
        print(f"Multilingual: {multilingual}")
        print("-" * 50)