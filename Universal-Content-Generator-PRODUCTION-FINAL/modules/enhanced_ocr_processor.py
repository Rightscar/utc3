"""
Enhanced OCR Processor Module
============================

Advanced OCR processing with language-specific models and confidence scoring.
Improves OCR accuracy for spiritual texts in multiple languages.

Features:
- Language-specific Tesseract models (Sanskrit, Hindi, Tamil, etc.)
- User control for OCR threshold tuning
- Confidence score logging for each page
- Automatic language detection
- OCR quality assessment and recommendations
"""

import os
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import streamlit as st

try:
    import pytesseract
    import pdf2image
    import fitz  # PyMuPDF
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# PIL Image import (needed for type hints)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    # Create a dummy Image class for type hints when PIL is not available
    class Image:
        class Image:
            pass

try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


@dataclass
class OCRResult:
    """Result of OCR processing for a single page"""
    page_number: int
    text: str
    confidence: float
    language: str
    method: str  # "standard", "ocr", "hybrid"
    processing_time: float
    warnings: List[str]
    char_count: int
    word_count: int


@dataclass
class OCRConfiguration:
    """OCR processing configuration"""
    language: str = "eng"
    psm: int = 6  # Page segmentation mode
    oem: int = 3  # OCR Engine mode
    confidence_threshold: float = 60.0
    dpi: int = 300
    preprocessing: bool = True
    custom_config: str = ""


class EnhancedOCRProcessor:
    """Enhanced OCR processor with language support and quality assessment"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Available language models
        self.language_models = {
            "eng": "English",
            "hin": "Hindi (हिन्दी)",
            "tam": "Tamil (தமிழ்)",
            "san": "Sanskrit (संस्कृत)",
            "ara": "Arabic (العربية)",
            "tha": "Thai (ไทย)",
            "tib": "Tibetan (བོད་ཡིག)",
            "jpn": "Japanese (日本語)",
            "chi_sim": "Chinese Simplified (简体中文)",
            "chi_tra": "Chinese Traditional (繁體中文)",
            "kor": "Korean (한국어)",
            "deu": "German (Deutsch)",
            "fra": "French (Français)",
            "spa": "Spanish (Español)",
            "ita": "Italian (Italiano)",
            "por": "Portuguese (Português)",
            "rus": "Russian (Русский)"
        }
        
        # Check available Tesseract languages
        self.available_languages = self._check_available_languages()
        
        # Default configuration
        self.default_config = OCRConfiguration()
    
    def _check_available_languages(self) -> List[str]:
        """Check which Tesseract language models are installed"""
        
        if not OCR_AVAILABLE:
            return []
        
        try:
            available = pytesseract.get_languages(config='')
            self.logger.info(f"Available Tesseract languages: {available}")
            return available
        except Exception as e:
            self.logger.error(f"Failed to check available languages: {e}")
            return ["eng"]  # Default fallback
    
    def detect_language(self, text: str) -> str:
        """Detect language of text content"""
        
        if not LANGDETECT_AVAILABLE or not text.strip():
            return "eng"
        
        try:
            detected = langdetect.detect(text)
            
            # Map common language codes to Tesseract codes
            language_mapping = {
                "en": "eng",
                "hi": "hin",
                "ta": "tam",
                "sa": "san",
                "ar": "ara",
                "th": "tha",
                "bo": "tib",
                "ja": "jpn",
                "zh-cn": "chi_sim",
                "zh-tw": "chi_tra",
                "ko": "kor",
                "de": "deu",
                "fr": "fra",
                "es": "spa",
                "it": "ita",
                "pt": "por",
                "ru": "rus"
            }
            
            tesseract_lang = language_mapping.get(detected, "eng")
            
            # Check if the detected language is available
            if tesseract_lang in self.available_languages:
                return tesseract_lang
            else:
                self.logger.warning(f"Detected language {tesseract_lang} not available, using English")
                return "eng"
                
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "eng"
    
    def preprocess_image(self, image: Image.Image, config: OCRConfiguration) -> Image.Image:
        """Preprocess image for better OCR accuracy"""
        
        if not config.preprocessing:
            return image
        
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast and sharpness
            from PIL import ImageEnhance, ImageFilter
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            self.logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text_with_confidence(self, 
                                   image: Image.Image, 
                                   config: OCRConfiguration) -> Tuple[str, float, List[str]]:
        """Extract text from image with confidence scoring"""
        
        if not OCR_AVAILABLE:
            return "", 0.0, ["OCR not available"]
        
        warnings = []
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image, config)
            
            # Build Tesseract configuration
            tesseract_config = f"--psm {config.psm} --oem {config.oem}"
            if config.custom_config:
                tesseract_config += f" {config.custom_config}"
            
            # Extract text with confidence data
            data = pytesseract.image_to_data(
                processed_image,
                lang=config.language,
                config=tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Calculate confidence and extract text
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Extract text
            text = pytesseract.image_to_string(
                processed_image,
                lang=config.language,
                config=tesseract_config
            )
            
            # Quality assessment
            if avg_confidence < config.confidence_threshold:
                warnings.append(f"Low OCR confidence: {avg_confidence:.1f}%")
            
            if len(text.strip()) < 50:
                warnings.append("Very little text extracted - possible scan quality issue")
            
            # Check for common OCR errors
            error_indicators = ['|||', '~~~', '```', 'lll', 'III']
            if any(indicator in text for indicator in error_indicators):
                warnings.append("Possible OCR artifacts detected")
            
            return text, avg_confidence, warnings
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            return "", 0.0, [f"OCR error: {str(e)}"]
    
    def process_pdf_page(self, 
                        pdf_path: str, 
                        page_num: int, 
                        config: OCRConfiguration) -> OCRResult:
        """Process a single PDF page with OCR"""
        
        import time
        start_time = time.time()
        
        warnings = []
        
        try:
            # First try standard text extraction
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            standard_text = page.get_text()
            
            # Check if standard extraction is sufficient
            if len(standard_text.strip()) > 100:
                # Good text extraction, no OCR needed
                processing_time = time.time() - start_time
                
                # Detect language for future reference
                detected_lang = self.detect_language(standard_text)
                
                return OCRResult(
                    page_number=page_num,
                    text=standard_text,
                    confidence=100.0,  # Standard extraction is 100% confident
                    language=detected_lang,
                    method="standard",
                    processing_time=processing_time,
                    warnings=[],
                    char_count=len(standard_text),
                    word_count=len(standard_text.split())
                )
            
            # Standard extraction insufficient, use OCR
            warnings.append("Standard text extraction insufficient, using OCR")
            
            # Convert page to image
            images = pdf2image.convert_from_path(
                pdf_path,
                first_page=page_num + 1,
                last_page=page_num + 1,
                dpi=config.dpi
            )
            
            if not images:
                raise Exception("Failed to convert PDF page to image")
            
            image = images[0]
            
            # Auto-detect language if not specified or if using default
            if config.language == "eng" and standard_text:
                detected_lang = self.detect_language(standard_text)
                if detected_lang != "eng" and detected_lang in self.available_languages:
                    config.language = detected_lang
                    warnings.append(f"Auto-detected language: {self.language_models.get(detected_lang, detected_lang)}")
            
            # Extract text with OCR
            ocr_text, confidence, ocr_warnings = self.extract_text_with_confidence(image, config)
            warnings.extend(ocr_warnings)
            
            # Combine with standard text if available
            if standard_text.strip() and ocr_text.strip():
                # Use hybrid approach - prefer OCR but supplement with standard
                final_text = ocr_text
                method = "hybrid"
                warnings.append("Used hybrid extraction (OCR + standard)")
            else:
                final_text = ocr_text or standard_text
                method = "ocr"
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                page_number=page_num,
                text=final_text,
                confidence=confidence,
                language=config.language,
                method=method,
                processing_time=processing_time,
                warnings=warnings,
                char_count=len(final_text),
                word_count=len(final_text.split())
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Failed to process page {page_num}: {str(e)}"
            self.logger.error(error_msg)
            
            return OCRResult(
                page_number=page_num,
                text="",
                confidence=0.0,
                language=config.language,
                method="error",
                processing_time=processing_time,
                warnings=[error_msg],
                char_count=0,
                word_count=0
            )
    
    def get_ocr_recommendations(self, results: List[OCRResult]) -> List[str]:
        """Get recommendations for improving OCR quality"""
        
        recommendations = []
        
        if not results:
            return ["No OCR results to analyze"]
        
        # Analyze confidence scores
        confidences = [r.confidence for r in results if r.confidence > 0]
        if confidences:
            avg_confidence = sum(confidences) / len(confidences)
            
            if avg_confidence < 50:
                recommendations.extend([
                    "Very low OCR confidence detected",
                    "Consider rescanning at higher resolution (600+ DPI)",
                    "Ensure good lighting and contrast in original scan",
                    "Try different page segmentation mode (PSM)"
                ])
            elif avg_confidence < 70:
                recommendations.extend([
                    "Moderate OCR confidence - some improvements possible",
                    "Consider preprocessing options or higher DPI",
                    "Check if correct language model is selected"
                ])
        
        # Analyze language consistency
        languages = [r.language for r in results]
        unique_languages = set(languages)
        if len(unique_languages) > 1:
            recommendations.append(f"Multiple languages detected: {', '.join(unique_languages)} - consider processing separately")
        
        # Analyze processing methods
        methods = [r.method for r in results]
        ocr_pages = sum(1 for m in methods if m in ["ocr", "hybrid"])
        if ocr_pages > len(results) * 0.8:
            recommendations.append("Most pages required OCR - document may be image-based or low quality")
        
        # Analyze warnings
        all_warnings = []
        for result in results:
            all_warnings.extend(result.warnings)
        
        common_warnings = {}
        for warning in all_warnings:
            common_warnings[warning] = common_warnings.get(warning, 0) + 1
        
        for warning, count in common_warnings.items():
            if count > len(results) * 0.3:  # Warning appears in >30% of pages
                recommendations.append(f"Common issue: {warning}")
        
        return recommendations
    
    def render_ocr_configuration(self) -> OCRConfiguration:
        """Render OCR configuration UI and return selected configuration"""
        
        with st.expander("⚙️ OCR Configuration", expanded=False):
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Language selection
                available_lang_options = {
                    code: f"{name} ({'✅' if code in self.available_languages else '❌'})"
                    for code, name in self.language_models.items()
                }
                
                selected_lang = st.selectbox(
                    "OCR Language",
                    options=list(available_lang_options.keys()),
                    format_func=lambda x: available_lang_options[x],
                    index=0,
                    help="Select the primary language for OCR processing"
                )
                
                if selected_lang not in self.available_languages:
                    st.warning(f"⚠️ {self.language_models[selected_lang]} model not installed")
                    st.info("Install with: `sudo apt-get install tesseract-ocr-{lang}`")
                
                # Page segmentation mode
                psm_options = {
                    6: "Uniform block of text (default)",
                    3: "Fully automatic page segmentation",
                    4: "Single column of text",
                    7: "Single text line",
                    8: "Single word",
                    11: "Sparse text",
                    12: "Sparse text with OSD",
                    13: "Raw line (bypass hOCR)"
                }
                
                psm = st.selectbox(
                    "Page Segmentation Mode",
                    options=list(psm_options.keys()),
                    format_func=lambda x: f"{x}: {psm_options[x]}",
                    index=0,
                    help="How Tesseract should segment the page"
                )
            
            with col2:
                # Confidence threshold
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0,
                    max_value=100.0,
                    value=60.0,
                    step=5.0,
                    help="Minimum confidence for accepting OCR results"
                )
                
                # DPI setting
                dpi = st.selectbox(
                    "Image DPI",
                    options=[150, 200, 300, 400, 600],
                    index=2,
                    help="Higher DPI improves accuracy but increases processing time"
                )
                
                # Preprocessing
                preprocessing = st.checkbox(
                    "Enable Preprocessing",
                    value=True,
                    help="Apply contrast enhancement and noise reduction"
                )
                
                # Custom configuration
                custom_config = st.text_input(
                    "Custom Tesseract Config",
                    value="",
                    help="Additional Tesseract parameters (advanced users)"
                )
            
            # Show available languages
            if st.checkbox("Show Available Language Models"):
                st.write("**Installed Language Models:**")
                for lang in self.available_languages:
                    name = self.language_models.get(lang, lang)
                    st.write(f"✅ {lang}: {name}")
                
                missing_langs = set(self.language_models.keys()) - set(self.available_languages)
                if missing_langs:
                    st.write("**Missing Language Models:**")
                    for lang in missing_langs:
                        name = self.language_models[lang]
                        st.write(f"❌ {lang}: {name}")
        
        return OCRConfiguration(
            language=selected_lang,
            psm=psm,
            confidence_threshold=confidence_threshold,
            dpi=dpi,
            preprocessing=preprocessing,
            custom_config=custom_config
        )
    
    def render_ocr_results_summary(self, results: List[OCRResult]):
        """Render summary of OCR processing results"""
        
        if not results:
            return
        
        with st.expander("📊 OCR Processing Summary", expanded=True):
            
            # Overall statistics
            total_pages = len(results)
            successful_pages = sum(1 for r in results if r.text.strip())
            avg_confidence = sum(r.confidence for r in results) / total_pages if total_pages else 0
            total_processing_time = sum(r.processing_time for r in results)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Pages", total_pages)
            with col2:
                st.metric("Successful", f"{successful_pages}/{total_pages}")
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
            with col4:
                st.metric("Processing Time", f"{total_processing_time:.1f}s")
            
            # Method distribution
            methods = [r.method for r in results]
            method_counts = {}
            for method in methods:
                method_counts[method] = method_counts.get(method, 0) + 1
            
            st.write("**Processing Methods:**")
            for method, count in method_counts.items():
                percentage = (count / total_pages) * 100
                st.write(f"- {method.title()}: {count} pages ({percentage:.1f}%)")
            
            # Language distribution
            languages = [r.language for r in results]
            unique_languages = set(languages)
            if len(unique_languages) > 1:
                st.write("**Languages Detected:**")
                for lang in unique_languages:
                    count = languages.count(lang)
                    percentage = (count / total_pages) * 100
                    name = self.language_models.get(lang, lang)
                    st.write(f"- {name}: {count} pages ({percentage:.1f}%)")
            
            # Quality assessment
            low_confidence_pages = [r for r in results if r.confidence < 60]
            if low_confidence_pages:
                st.warning(f"⚠️ {len(low_confidence_pages)} pages have low confidence (<60%)")
            
            # Recommendations
            recommendations = self.get_ocr_recommendations(results)
            if recommendations:
                st.write("**Recommendations:**")
                for rec in recommendations:
                    st.write(f"- {rec}")


# Global enhanced OCR processor instance
enhanced_ocr_processor = EnhancedOCRProcessor()


def get_enhanced_ocr_processor() -> EnhancedOCRProcessor:
    """Get the global enhanced OCR processor instance"""
    return enhanced_ocr_processor

