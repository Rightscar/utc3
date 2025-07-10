"""
Multilingual NLP Processor
=========================

This module extends our NLP capabilities to support multiple languages,
providing language detection, cross-lingual processing, and multilingual
semantic understanding for global content processing.

Features:
- Automatic language detection
- Multi-language spaCy model support
- Cross-lingual semantic similarity
- Language-specific processing optimization
- Translation integration capabilities
- Multilingual quality assessment
"""

import logging
import spacy
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

# Language detection
try:
    from langdetect import detect, detect_langs, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("⚠️ langdetect not available, using basic language detection")

# Translation capabilities
try:
    from googletrans import Translator
    TRANSLATION_AVAILABLE = True
except ImportError:
    TRANSLATION_AVAILABLE = False
    logging.warning("⚠️ googletrans not available, translation features disabled")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LanguageInfo:
    """Information about detected language"""
    code: str
    name: str
    confidence: float
    script: str
    family: str

@dataclass
class MultilingualText:
    """Text with multilingual processing information"""
    text: str
    language: LanguageInfo
    original_language: Optional[str] = None
    translated_text: Optional[str] = None
    processing_notes: List[str] = None

class MultilingualProcessor:
    """Advanced multilingual NLP processing engine"""
    
    # Supported language configurations
    LANGUAGE_MODELS = {
        'en': 'en_core_web_sm',
        'es': 'es_core_news_sm',
        'fr': 'fr_core_news_sm', 
        'de': 'de_core_news_sm',
        'it': 'it_core_news_sm',
        'pt': 'pt_core_news_sm',
        'nl': 'nl_core_news_sm',
        'zh': 'zh_core_web_sm',
        'ja': 'ja_core_news_sm',
        'ru': 'ru_core_news_sm'
    }
    
    LANGUAGE_NAMES = {
        'en': 'English',
        'es': 'Spanish', 
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'nl': 'Dutch',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ru': 'Russian',
        'ar': 'Arabic',
        'hi': 'Hindi',
        'ko': 'Korean'
    }
    
    LANGUAGE_FAMILIES = {
        'en': 'Germanic',
        'es': 'Romance',
        'fr': 'Romance', 
        'de': 'Germanic',
        'it': 'Romance',
        'pt': 'Romance',
        'nl': 'Germanic',
        'zh': 'Sino-Tibetan',
        'ja': 'Japonic',
        'ru': 'Slavic',
        'ar': 'Semitic',
        'hi': 'Indo-Aryan',
        'ko': 'Koreanic'
    }
    
    SCRIPTS = {
        'en': 'Latin',
        'es': 'Latin',
        'fr': 'Latin',
        'de': 'Latin', 
        'it': 'Latin',
        'pt': 'Latin',
        'nl': 'Latin',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ru': 'Cyrillic',
        'ar': 'Arabic',
        'hi': 'Devanagari',
        'ko': 'Hangul'
    }
    
    def __init__(self, default_language: str = 'en', enable_translation: bool = True):
        """Initialize multilingual processor"""
        self.default_language = default_language
        self.enable_translation = enable_translation and TRANSLATION_AVAILABLE
        self.nlp_models = {}
        self.translator = None
        
        # Initialize translator
        if self.enable_translation:
            try:
                self.translator = Translator()
                logger.info("✅ Translation service initialized")
            except Exception as e:
                logger.warning(f"⚠️ Translation service failed to initialize: {e}")
                self.enable_translation = False
        
        # Load default language model
        self._load_language_model(default_language)
        
        logger.info(f"✅ Multilingual processor initialized with default language: {default_language}")
    
    def _load_language_model(self, language_code: str) -> bool:
        """Load spaCy model for specific language"""
        if language_code in self.nlp_models:
            return True
        
        model_name = self.LANGUAGE_MODELS.get(language_code)
        if not model_name:
            logger.warning(f"⚠️ No spaCy model available for language: {language_code}")
            return False
        
        try:
            self.nlp_models[language_code] = spacy.load(model_name)
            logger.info(f"✅ Loaded spaCy model for {language_code}: {model_name}")
            return True
        except OSError:
            logger.warning(f"⚠️ spaCy model not found: {model_name}")
            return False
    
    def detect_language(self, text: str) -> LanguageInfo:
        """Detect language of input text"""
        if not LANGDETECT_AVAILABLE:
            return self._basic_language_detection(text)
        
        try:
            # Use langdetect for accurate detection
            detected_langs = detect_langs(text)
            primary_lang = detected_langs[0]
            
            language_code = primary_lang.lang
            confidence = primary_lang.prob
            
            return LanguageInfo(
                code=language_code,
                name=self.LANGUAGE_NAMES.get(language_code, language_code.upper()),
                confidence=confidence,
                script=self.SCRIPTS.get(language_code, 'Unknown'),
                family=self.LANGUAGE_FAMILIES.get(language_code, 'Unknown')
            )
            
        except LangDetectException:
            logger.warning("⚠️ Language detection failed, using basic detection")
            return self._basic_language_detection(text)
    
    def _basic_language_detection(self, text: str) -> LanguageInfo:
        """Basic language detection using character patterns"""
        # Simple heuristics for common languages
        if re.search(r'[а-яё]', text.lower()):
            return LanguageInfo('ru', 'Russian', 0.8, 'Cyrillic', 'Slavic')
        elif re.search(r'[一-龯]', text):
            return LanguageInfo('zh', 'Chinese', 0.8, 'Chinese', 'Sino-Tibetan')
        elif re.search(r'[ひらがなカタカナ]', text):
            return LanguageInfo('ja', 'Japanese', 0.8, 'Japanese', 'Japonic')
        elif re.search(r'[가-힣]', text):
            return LanguageInfo('ko', 'Korean', 0.8, 'Hangul', 'Koreanic')
        elif re.search(r'[ا-ي]', text):
            return LanguageInfo('ar', 'Arabic', 0.8, 'Arabic', 'Semitic')
        elif re.search(r'[à-ÿ]', text.lower()):
            # Romance languages (rough heuristic)
            if 'ñ' in text.lower():
                return LanguageInfo('es', 'Spanish', 0.7, 'Latin', 'Romance')
            elif 'ç' in text.lower():
                return LanguageInfo('fr', 'French', 0.7, 'Latin', 'Romance')
            else:
                return LanguageInfo('fr', 'French', 0.6, 'Latin', 'Romance')
        else:
            # Default to English
            return LanguageInfo('en', 'English', 0.6, 'Latin', 'Germanic')
    
    def process_multilingual_text(self, text: str, target_language: Optional[str] = None) -> MultilingualText:
        """Process text with multilingual capabilities"""
        # Detect language
        detected_lang = self.detect_language(text)
        processing_notes = []
        
        # Determine processing language
        if target_language:
            processing_lang = target_language
        elif detected_lang.code in self.LANGUAGE_MODELS:
            processing_lang = detected_lang.code
        else:
            processing_lang = self.default_language
            processing_notes.append(f"No model for {detected_lang.name}, using {self.LANGUAGE_NAMES[processing_lang]}")
        
        # Load appropriate model
        if not self._load_language_model(processing_lang):
            processing_lang = self.default_language
            processing_notes.append(f"Fallback to {self.LANGUAGE_NAMES[processing_lang]}")
        
        # Handle translation if needed
        translated_text = None
        if (detected_lang.code != processing_lang and 
            self.enable_translation and 
            detected_lang.confidence > 0.8):
            
            translated_text = self.translate_text(text, detected_lang.code, processing_lang)
            if translated_text:
                processing_notes.append(f"Translated from {detected_lang.name} to {self.LANGUAGE_NAMES[processing_lang]}")
        
        return MultilingualText(
            text=text,
            language=detected_lang,
            original_language=detected_lang.code if translated_text else None,
            translated_text=translated_text,
            processing_notes=processing_notes
        )
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Translate text between languages"""
        if not self.enable_translation:
            return None
        
        try:
            result = self.translator.translate(text, src=source_lang, dest=target_lang)
            return result.text
        except Exception as e:
            logger.warning(f"⚠️ Translation failed: {e}")
            return None
    
    def get_nlp_model(self, language_code: str):
        """Get spaCy model for specific language"""
        if language_code not in self.nlp_models:
            if not self._load_language_model(language_code):
                # Fallback to default language
                language_code = self.default_language
        
        return self.nlp_models.get(language_code)
    
    def process_with_language_model(self, text: str, language_code: str) -> Any:
        """Process text with specific language model"""
        nlp = self.get_nlp_model(language_code)
        if nlp:
            return nlp(text)
        else:
            logger.warning(f"⚠️ No model available for {language_code}")
            return None
    
    def extract_multilingual_entities(self, text: str) -> Dict[str, Any]:
        """Extract entities with multilingual support"""
        multilingual_text = self.process_multilingual_text(text)
        
        # Use appropriate text for processing
        processing_text = multilingual_text.translated_text or multilingual_text.text
        processing_lang = multilingual_text.language.code
        
        # Get appropriate model
        nlp = self.get_nlp_model(processing_lang)
        if not nlp:
            return {'entities': [], 'language': multilingual_text.language.code, 'error': 'No model available'}
        
        # Process text
        doc = nlp(processing_text)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_) if spacy.explain(ent.label_) else ent.label_
            })
        
        return {
            'entities': entities,
            'language': multilingual_text.language.code,
            'language_name': multilingual_text.language.name,
            'confidence': multilingual_text.language.confidence,
            'translated': multilingual_text.translated_text is not None,
            'processing_notes': multilingual_text.processing_notes
        }
    
    def get_supported_languages(self) -> Dict[str, Dict[str, str]]:
        """Get information about supported languages"""
        supported = {}
        
        for code, model in self.LANGUAGE_MODELS.items():
            supported[code] = {
                'name': self.LANGUAGE_NAMES.get(code, code.upper()),
                'model': model,
                'family': self.LANGUAGE_FAMILIES.get(code, 'Unknown'),
                'script': self.SCRIPTS.get(code, 'Unknown'),
                'loaded': code in self.nlp_models
            }
        
        return supported
    
    def get_language_statistics(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze language distribution in text collection"""
        language_counts = Counter()
        confidence_scores = []
        
        for text in texts:
            lang_info = self.detect_language(text)
            language_counts[lang_info.code] += 1
            confidence_scores.append(lang_info.confidence)
        
        # Calculate statistics
        total_texts = len(texts)
        language_distribution = {}
        
        for lang_code, count in language_counts.items():
            percentage = (count / total_texts) * 100
            language_distribution[lang_code] = {
                'name': self.LANGUAGE_NAMES.get(lang_code, lang_code.upper()),
                'count': count,
                'percentage': percentage
            }
        
        return {
            'total_texts': total_texts,
            'languages_detected': len(language_counts),
            'language_distribution': language_distribution,
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0,
            'most_common_language': language_counts.most_common(1)[0] if language_counts else None
        }
    
    def optimize_for_language(self, language_code: str) -> bool:
        """Optimize processor for specific language"""
        if self._load_language_model(language_code):
            self.default_language = language_code
            logger.info(f"✅ Optimized for language: {self.LANGUAGE_NAMES.get(language_code, language_code)}")
            return True
        return False

# Example usage and testing
if __name__ == "__main__":
    # Test multilingual processor
    processor = MultilingualProcessor()
    
    test_texts = [
        "Hello, this is a test in English.",
        "Hola, esto es una prueba en español.",
        "Bonjour, ceci est un test en français.",
        "Hallo, das ist ein Test auf Deutsch.",
        "Привет, это тест на русском языке.",
        "こんにちは、これは日本語のテストです。"
    ]
    
    print("🌍 Testing Multilingual Processor...")
    
    # Test language detection
    for text in test_texts:
        lang_info = processor.detect_language(text)
        print(f"📝 '{text[:30]}...' -> {lang_info.name} ({lang_info.confidence:.2f})")
    
    # Test multilingual processing
    multilingual_result = processor.process_multilingual_text(test_texts[1])  # Spanish text
    print(f"🔄 Multilingual processing: {multilingual_result}")
    
    # Test entity extraction
    entities = processor.extract_multilingual_entities("Barack Obama was the President of the United States.")
    print(f"🏷️ Entities extracted: {len(entities['entities'])} entities")
    
    # Test language statistics
    stats = processor.get_language_statistics(test_texts)
    print(f"📊 Language statistics: {stats['languages_detected']} languages detected")
    
    # Test supported languages
    supported = processor.get_supported_languages()
    print(f"🌐 Supported languages: {len(supported)} languages available")
    
    print("🎉 Multilingual processor test completed!")

