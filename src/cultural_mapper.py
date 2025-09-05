"""
Katha Cultural Context Mappings
Authentic Indian emotional expressions for Bengali and Hindi
"""

from typing import Dict, List, Tuple, Optional


class KathaCulturalMappings:
    """
    Cultural emotion mappings for authentic Indian language processing
    This captures the nuanced emotional expressions unique to Bengali and Hindi cultures
    """
    
    def __init__(self):
        self.cultural_contexts = {
            'bengali': {
                'emotions': {
                    'joy': {
                        'words': ['খুশি', 'আনন্দ', 'আহ্লাদ', 'হর্ষ', 'উল্লাস', 'প্রফুল্ল'],
                        'expressions': ['দারুণ', 'অসাধারণ', 'চমৎকার', 'বাহ', 'সুন্দর'],
                        'intensity_markers': ['খুব', 'অনেক', 'বেশ', 'ভীষণ', 'অত্যন্ত'],
                        'cultural_phrases': ['মন ভালো', 'খুশিতে আত্মহারা', 'আনন্দে উদ্বেল'],
                        'prosody': {
                            'pitch_range': (150, 280),
                            'speed_factor': 1.2,
                            'energy': 1.4,
                            'rhythm': 'melodic'
                        }
                    },
                    'sadness': {
                        'words': ['দুঃখ', 'কষ্ট', 'বেদনা', 'ব্যথা', 'বিষণ্ণতা', 'হতাশা'],
                        'expressions': ['মন খারাপ', 'দুঃখিত', 'বিষণ্ণ', 'উদাস'],
                        'intensity_markers': ['খুব', 'অনেক', 'গভীর', 'তীব্র'],
                        'cultural_phrases': ['মন কেমন করছে', 'হৃদয় ভারাক্রান্ত', 'চোখে জল'],
                        'prosody': {
                            'pitch_range': (80, 150),
                            'speed_factor': 0.7,
                            'energy': 0.6,
                            'rhythm': 'slow_melodic'
                        }
                    },
                    'anger': {
                        'words': ['রাগ', 'ক্রোধ', 'ঘৃণা', 'বিরক্তি', 'অসন্তুষ্টি'],
                        'expressions': ['রেগে গেছি', 'ক্ষেপে গেছি', 'মাথা গরম'],
                        'intensity_markers': ['খুব', 'ভীষণ', 'প্রচণ্ড'],
                        'cultural_phrases': ['রাগে অগ্নিশর্মা', 'চোখ লাল', 'মাথা ঠাণ্ডা করো'],
                        'prosody': {
                            'pitch_range': (120, 200),
                            'speed_factor': 1.3,
                            'energy': 1.5,
                            'rhythm': 'sharp'
                        }
                    },
                    'surprise': {
                        'words': ['আশ্চর্য', 'অবাক', 'চমক', 'বিস্ময়', 'হতবাক'],
                        'expressions': ['বাহ', 'অবিশ্বাস্য', 'এ কি!', 'সত্যিই!'],
                        'intensity_markers': ['খুব', 'অনেক', 'একেবারে'],
                        'cultural_phrases': ['চোখ কপালে', 'মুখ হা', 'অবাক কাণ্ড'],
                        'prosody': {
                            'pitch_range': (180, 300),
                            'speed_factor': 1.1,
                            'energy': 1.3,
                            'rhythm': 'ascending'
                        }
                    }
                },
                'formality_levels': {
                    'high_respect': {
                        'pronouns': ['আপনি', 'তিনি'],
                        'titles': ['দাদা', 'দিদি', 'মা', 'বাবা', 'কাকা', 'কাকিমা', 'মামা', 'মামি'],
                        'honorifics': ['জী', 'স্যার', 'ম্যাডাম'],
                        'phrases': ['দয়া করে', 'অনুগ্রহ করে', 'আপনার কৃপা'],
                        'prosody_modifier': {
                            'speed_factor': 0.9,
                            'pitch_lower': 0.8,
                            'energy': 0.8
                        }
                    },
                    'medium_respect': {
                        'pronouns': ['তুমি'],
                        'phrases': ['প্লিজ', 'একটু'],
                        'prosody_modifier': {
                            'speed_factor': 1.0,
                            'pitch_lower': 1.0,
                            'energy': 1.0
                        }
                    },
                    'casual': {
                        'pronouns': ['তুই'],
                        'phrases': ['ইয়ে', 'এই', 'আরে'],
                        'prosody_modifier': {
                            'speed_factor': 1.2,
                            'pitch_lower': 1.1,
                            'energy': 1.2
                        }
                    }
                },
                'regional_variations': {
                    'kolkata': {
                        'enthusiasm_words': ['দারুণ', 'ফাটাফাটি', 'জোস'],
                        'casual_expressions': ['কি যে বলিস', 'ব্যাপার কি'],
                        'emotional_intensifiers': ['একেবারে', 'সম্পূর্ণ']
                    },
                    'dhaka': {
                        'enthusiasm_words': ['অসাধারণ', 'চমৎকার'],
                        'casual_expressions': ['কি অবস্থা', 'কেমন আছ'],
                        'emotional_intensifiers': ['পুরাপুরি', 'সম্পূর্ণভাবে']
                    }
                }
            },
            
            'hindi': {
                'emotions': {
                    'joy': {
                        'words': ['खुश', 'खुशी', 'आनंद', 'प्रसन्न', 'हर्षित', 'प्रफुल्लित'],
                        'expressions': ['बहुत अच्छा', 'वाह', 'शानदार', 'कमाल', 'लाजवाब'],
                        'intensity_markers': ['बहुत', 'काफी', 'अत्यधिक', 'बेहद'],
                        'cultural_phrases': ['दिल खुश', 'मन प्रसन्न', 'खुशी से झूम उठा'],
                        'prosody': {
                            'pitch_range': (140, 250),
                            'speed_factor': 1.1,
                            'energy': 1.3,
                            'rhythm': 'rhythmic'
                        }
                    },
                    'sadness': {
                        'words': ['दुःख', 'गम', 'शोक', 'उदासी', 'निराशा', 'पीड़ा'],
                        'expressions': ['दुःखी', 'उदास', 'परेशान', 'निराश'],
                        'intensity_markers': ['बहुत', 'काफी', 'गहरा', 'तीव्र'],
                        'cultural_phrases': ['दिल टूटा', 'मन भारी', 'आंखों में आंसू'],
                        'prosody': {
                            'pitch_range': (90, 140),
                            'speed_factor': 0.8,
                            'energy': 0.7,
                            'rhythm': 'slow_steady'
                        }
                    },
                    'anger': {
                        'words': ['गुस्सा', 'क्रोध', 'नाराजी', 'चिढ़', 'आक्रोश'],
                        'expressions': ['गुस्से में', 'नाराज़', 'चिढ़ गया', 'परेशान'],
                        'intensity_markers': ['बहुत', 'बेहद', 'अत्यधिक'],
                        'cultural_phrases': ['सिर गर्म', 'आग बबूला', 'खून खौल रहा'],
                        'prosody': {
                            'pitch_range': (130, 190),
                            'speed_factor': 1.3,
                            'energy': 1.4,
                            'rhythm': 'forceful'
                        }
                    },
                    'surprise': {
                        'words': ['आश्चर्य', 'हैरानी', 'अचरज', 'चौंक'],
                        'expressions': ['वाह', 'अरे', 'ये क्या', 'सच में'],
                        'intensity_markers': ['बहुत', 'काफी', 'पूरी तरह'],
                        'cultural_phrases': ['होश उड़ गए', 'आंखें फटी रह गईं', 'दंग रह गया'],
                        'prosody': {
                            'pitch_range': (170, 280),
                            'speed_factor': 1.05,
                            'energy': 1.2,
                            'rhythm': 'rising'
                        }
                    }
                },
                'formality_levels': {
                    'high_respect': {
                        'pronouns': ['आप', 'आपका'],
                        'titles': ['जी', 'साहब', 'श्रीमान', 'श्रीमती', 'सर', 'मैडम'],
                        'honorifics': ['जी हां', 'जी नहीं', 'आदरणीय'],
                        'phrases': ['कृपया', 'मेहरबानी', 'आपकी कृपा'],
                        'prosody_modifier': {
                            'speed_factor': 0.85,
                            'pitch_lower': 0.9,
                            'energy': 0.8
                        }
                    },
                    'medium_respect': {
                        'pronouns': ['तुम', 'तुम्हारा'],
                        'phrases': ['प्लीज़', 'जरा', 'थोड़ा'],
                        'prosody_modifier': {
                            'speed_factor': 1.0,
                            'pitch_lower': 1.0,
                            'energy': 1.0
                        }
                    },
                    'casual': {
                        'pronouns': ['तू', 'तेरा'],
                        'phrases': ['यार', 'भाई', 'अरे'],
                        'prosody_modifier': {
                            'speed_factor': 1.15,
                            'pitch_lower': 1.1,
                            'energy': 1.1
                        }
                    }
                },
                'regional_variations': {
                    'delhi': {
                        'enthusiasm_words': ['बिल्कुल', 'एकदम', 'घंटा'],
                        'casual_expressions': ['यार', 'भाई', 'क्या बात है'],
                        'emotional_intensifiers': ['पूरी तरह', 'बिल्कुल']
                    },
                    'mumbai': {
                        'enthusiasm_words': ['बम्बई', 'मस्त', 'झकास'],
                        'casual_expressions': ['क्या रे', 'चल ना'],
                        'emotional_intensifiers': ['एकदम', 'बिल्कुल']
                    },
                    'punjab': {
                        'enthusiasm_words': ['वाह', 'शाबाश', 'बहुत बढ़िया'],
                        'casual_expressions': ['यार', 'भाई जी'],
                        'emotional_intensifiers': ['पूरी तरह', 'बिल्कुल ठीक']
                    }
                }
            }
        }
        
        # Conversation context patterns
        self.conversation_patterns = {
            'greeting_responses': {
                'bengali': {
                    'morning': 'সুপ্রভাত! আজ কেমন লাগছে?',
                    'general': 'নমস্কার! কেমন আছেন?',
                    'evening': 'শুভ সন্ধ্যা! দিনটা কেমন কাটল?'
                },
                'hindi': {
                    'morning': 'सुप्रभात! आज कैसा लग रहा है?',
                    'general': 'नमस्ते! कैसे हैं आप?',
                    'evening': 'शुभ संध्या! दिन कैसा रहा?'
                }
            },
            'empathy_responses': {
                'bengali': {
                    'joy': 'আমিও আপনার সাথে খুশি! চমৎকার!',
                    'sadness': 'আমি বুঝতে পারছি। সব ঠিক হয়ে যাবে।',
                    'anger': 'আমি আপনার রাগের কারণ বুঝতে পারছি। শান্ত হোন।',
                    'fear': 'চিন্তা করবেন না। আমি আছি।',
                    'surprise': 'সত্যিই অবাক করার মতো!'
                },
                'hindi': {
                    'joy': 'मुझे भी बहुत खुशी हो रही है! शानदार!',
                    'sadness': 'मैं समझ सकता हूं। सब ठीक हो जाएगा।',
                    'anger': 'मैं आपकी परेशानी समझ सकता हूं। शांत हो जाइए।',
                    'fear': 'चिंता मत कीजिए। मैं हूं न।',
                    'surprise': 'वाकई चौंकाने वाली बात है!'
                }
            }
        }
    
    def detect_cultural_context(self, text: str, language: str) -> Dict:
        """
        Detect cultural and emotional context from text
        """
        if language not in ['bn', 'hi']:
            return {'formality': 'medium', 'region': 'standard', 'emotion_intensity': 1.0}
        
        lang_key = 'bengali' if language == 'bn' else 'hindi'
        lang_data = self.cultural_contexts[lang_key]
        
        context = {
            'formality': 'medium',
            'region': 'standard',
            'emotion_intensity': 1.0,
            'cultural_markers': []
        }
        
        # Detect formality level
        text_lower = text.lower()
        
        for level, markers in lang_data['formality_levels'].items():
            for pronoun in markers.get('pronouns', []):
                if pronoun.lower() in text_lower:
                    context['formality'] = level
                    break
            
            for title in markers.get('titles', []):
                if title.lower() in text_lower:
                    context['formality'] = level
                    context['cultural_markers'].append(f"respectful_title: {title}")
                    break
        
        # Detect regional variations
        for region, markers in lang_data.get('regional_variations', {}).items():
            for word in markers.get('enthusiasm_words', []):
                if word.lower() in text_lower:
                    context['region'] = region
                    context['cultural_markers'].append(f"regional: {region}")
                    break
        
        # Detect emotion intensity
        for emotion_type, emotion_data in lang_data['emotions'].items():
            intensity_markers = emotion_data.get('intensity_markers', [])
            for marker in intensity_markers:
                if marker.lower() in text_lower:
                    context['emotion_intensity'] *= 1.3
                    context['cultural_markers'].append(f"intensity: {marker}")
        
        return context
    
    def get_cultural_response(self, user_emotion: str, language: str, context: Dict) -> str:
        """
        Generate culturally appropriate response
        """
        lang_key = 'bengali' if language == 'bn' else 'hindi' if language == 'hi' else None
        
        if not lang_key or lang_key not in self.conversation_patterns['empathy_responses']:
            return "I understand your feelings."
        
        responses = self.conversation_patterns['empathy_responses'][lang_key]
        base_response = responses.get(user_emotion, responses.get('sadness'))  # Default to empathetic
        
        # Modify response based on formality
        if context.get('formality') == 'high_respect':
            if language == 'bn':
                base_response = base_response.replace('তুমি', 'আপনি').replace('তোমার', 'আপনার')
            elif language == 'hi':
                base_response = base_response.replace('तुम', 'आप').replace('तुम्हारा', 'आपका')
        
        return base_response
    
    def get_prosody_config(self, emotion: str, language: str, cultural_context: Dict) -> Dict:
        """
        Get prosodic configuration for emotional speech synthesis
        """
        lang_key = 'bengali' if language == 'bn' else 'hindi' if language == 'hi' else None
        
        if not lang_key or lang_key not in self.cultural_contexts:
            return {'speed_factor': 1.0, 'pitch_shift': 0.0, 'energy': 1.0}
        
        # Get base emotion config
        emotion_config = self.cultural_contexts[lang_key]['emotions'].get(
            emotion, 
            self.cultural_contexts[lang_key]['emotions']['joy']  # Default
        )
        
        prosody = emotion_config['prosody'].copy()
        
        # Apply formality modifications
        formality = cultural_context.get('formality', 'medium')
        if formality in self.cultural_contexts[lang_key]['formality_levels']:
            formality_modifier = self.cultural_contexts[lang_key]['formality_levels'][formality]['prosody_modifier']
            
            prosody['speed_factor'] *= formality_modifier.get('speed_factor', 1.0)
            prosody['energy'] *= formality_modifier.get('energy', 1.0)
            if 'pitch_range' in prosody:
                low, high = prosody['pitch_range']
                pitch_modifier = formality_modifier.get('pitch_lower', 1.0)
                prosody['pitch_range'] = (low * pitch_modifier, high * pitch_modifier)
        
        # Apply emotion intensity
        intensity = cultural_context.get('emotion_intensity', 1.0)
        prosody['speed_factor'] = 1.0 + (prosody['speed_factor'] - 1.0) * min(intensity, 1.5)
        prosody['energy'] *= min(intensity, 1.5)
        
        return prosody
    
    def get_cultural_emotion_words(self, language: str, emotion: str) -> List[str]:
        """Get culture-specific emotion words"""
        lang_key = 'bengali' if language == 'bn' else 'hindi' if language == 'hi' else None
        
        if not lang_key or lang_key not in self.cultural_contexts:
            return []
        
        emotion_data = self.cultural_contexts[lang_key]['emotions'].get(emotion, {})
        return emotion_data.get('words', []) + emotion_data.get('expressions', [])
    
    def is_respectful_context(self, text: str, language: str) -> bool:
        """Check if text contains respectful/formal language"""
        context = self.detect_cultural_context(text, language)
        return context['formality'] == 'high_respect'
    
    def get_greeting_response(self, time_of_day: str, language: str) -> str:
        """Get culturally appropriate greeting"""
        lang_key = 'bengali' if language == 'bn' else 'hindi' if language == 'hi' else None
        
        if not lang_key or lang_key not in self.conversation_patterns['greeting_responses']:
            return "Hello!"
        
        greetings = self.conversation_patterns['greeting_responses'][lang_key]
        return greetings.get(time_of_day, greetings['general'])


# Example usage and testing
if __name__ == "__main__":
    cultural_mapper = KathaCulturalMappings()
    
    # Test Bengali cultural context
    bengali_text = "আপনি কেমন আছেন দাদা? আমি খুব খুশি!"
    context = cultural_mapper.detect_cultural_context(bengali_text, 'bn')
    print(f"Bengali Context: {context}")
    
    response = cultural_mapper.get_cultural_response('joy', 'bn', context)
    print(f"Bengali Response: {response}")
    
    prosody = cultural_mapper.get_prosody_config('joy', 'bn', context)
    print(f"Bengali Prosody: {prosody}")
    
    print("\n" + "="*50 + "\n")
    
    # Test Hindi cultural context
    hindi_text = "आप कैसे हैं साहब? मैं बहुत परेशान हूं।"
    context = cultural_mapper.detect_cultural_context(hindi_text, 'hi')
    print(f"Hindi Context: {context}")
    
    response = cultural_mapper.get_cultural_response('sadness', 'hi', context)
    print(f"Hindi Response: {response}")
    
    prosody = cultural_mapper.get_prosody_config('sadness', 'hi', context)
    print(f"Hindi Prosody: {prosody}")
    
    print("\n" + "="*50 + "\n")
    
    # Test emotion words
    bn_joy_words = cultural_mapper.get_cultural_emotion_words('bn', 'joy')
    print(f"Bengali Joy Words: {bn_joy_words}")
    
    hi_anger_words = cultural_mapper.get_cultural_emotion_words('hi', 'anger')
    print(f"Hindi Anger Words: {hi_anger_words}")
    
    # Test greetings
    morning_greeting_bn = cultural_mapper.get_greeting_response('morning', 'bn')
    evening_greeting_hi = cultural_mapper.get_greeting_response('evening', 'hi')
    print(f"Bengali Morning Greeting: {morning_greeting_bn}")
    print(f"Hindi Evening Greeting: {evening_greeting_hi}")