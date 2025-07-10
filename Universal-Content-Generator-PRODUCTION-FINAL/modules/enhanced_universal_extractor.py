"""
Enhanced Universal Extractor
============================

Enhanced version with smart Q&A vs monologue detection.
Implements regex-based structured Q&A detection and handles standalone content appropriately.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedUniversalExtractor:
    """
    Enhanced universal content extractor with smart Q&A vs monologue detection
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.txt', '.docx', '.epub', '.md']
        
        # Enhanced Q&A detection patterns
        self.qa_patterns = [
            # Pattern 1: Q: ... A: ... (most common)
            r'Q[:\-]\s*(.+?)\s*A[:\-]\s*(.+?)(?=Q[:\-]|$)',
            # Pattern 2: Question: ... Answer: ...
            r'Question[:\-]\s*(.+?)\s*Answer[:\-]\s*(.+?)(?=Question[:\-]|$)',
            # Pattern 3: Questioner: ... Teacher: ...
            r'(?:Questioner|Student|Seeker)[:\-]\s*(.+?)\s*(?:Teacher|Master|Guru)[:\-]\s*(.+?)(?=(?:Questioner|Student|Seeker)[:\-]|$)',
            # Pattern 4: Numbered Q&A
            r'\d+\.\s*Q[:\-]\s*(.+?)\s*A[:\-]\s*(.+?)(?=\d+\.\s*Q[:\-]|$)',
        ]
        
        # Dialogue detection patterns (speaker-based)
        self.dialogue_patterns = [
            r'^([A-Z][^:]*?):\s*(.+)$',  # Speaker: content
            r'^([A-Z][^—]*?)—\s*(.+)$',  # Speaker— content
            r'"([^"]+)"\s*(?:said|asked|replied)\s*([^,]+)',  # Quoted speech
        ]
        
        self.extraction_stats = {
            'total_processed': 0,
            'qa_detected': 0,
            'dialogue_detected': 0,
            'monologue_processed': 0,
            'extraction_errors': 0
        }
    
    def extract_text(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Extract text content from any supported file format with smart detection
        
        Args:
            file_path: Path to the file to extract
            
        Returns:
            List of extracted examples with metadata
        """
        self.extraction_stats['total_processed'] += 1
        
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.pdf':
                raw_content = self._extract_pdf(file_path)
            elif file_ext == '.txt':
                raw_content = self._extract_txt(file_path)
            elif file_ext == '.docx':
                raw_content = self._extract_docx(file_path)
            elif file_ext == '.epub':
                raw_content = self._extract_epub(file_path)
            elif file_ext == '.md':
                raw_content = self._extract_md(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Smart content detection and processing
            return self._smart_content_detection(raw_content, file_ext)
                
        except Exception as e:
            logger.error(f"Extraction error for {file_path}: {e}")
            self.extraction_stats['extraction_errors'] += 1
            return []
    
    def extract_content(self, uploaded_file) -> str:
        """
        Extract content from uploaded file object (for Streamlit compatibility)
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Extracted text content as string
        """
        # Validate input
        if uploaded_file is None:
            logger.warning("extract_content called with None uploaded_file")
            return ""
        
        if not hasattr(uploaded_file, 'name') or not hasattr(uploaded_file, 'read'):
            logger.warning("extract_content called with invalid uploaded_file object")
            return ""
        
        import tempfile
        
        try:
            # Save uploaded file to temporary path
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            
            # Extract using file path method
            extracted_data = self.extract_text(tmp_file_path)
            
            # Clean up temporary file
            import os
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
            # Combine all extracted content into a single string
            if extracted_data:
                combined_text = "\n\n".join([
                    f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}" 
                    if item.get('type') in ['qa', 'dialogue'] 
                    else item.get('answer', '') or item.get('question', '')
                    for item in extracted_data
                ])
                return combined_text
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Content extraction error for {uploaded_file.name}: {e}")
            return ""
    
    def _smart_content_detection(self, text: str, source_type: str) -> List[Dict[str, Any]]:
        """
        Smart detection of content type and appropriate processing
        
        Args:
            text: Raw extracted text
            source_type: Type of source file
            
        Returns:
            List of processed examples with appropriate handling
        """
        examples = []
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Step 1: Try structured Q&A detection
        qa_examples = self._detect_structured_qa(text)
        if qa_examples:
            self.extraction_stats['qa_detected'] += 1
            examples.extend(qa_examples)
            logger.info(f"Detected structured Q&A: {len(qa_examples)} pairs")
        
        # Step 2: Try dialogue detection
        if not examples:
            dialogue_examples = self._detect_dialogue_format(text)
            if dialogue_examples:
                self.extraction_stats['dialogue_detected'] += 1
                examples.extend(dialogue_examples)
                logger.info(f"Detected dialogue format: {len(dialogue_examples)} exchanges")
        
        # Step 3: Process as monologue/standalone content
        if not examples:
            self.extraction_stats['monologue_processed'] += 1
            monologue_examples = self._process_monologue_content(text)
            examples.extend(monologue_examples)
            logger.info(f"Processed as monologue: {len(monologue_examples)} passages")
        
        # Add metadata to all examples
        for example in examples:
            example.update({
                'source_type': source_type,
                'extraction_method': 'enhanced_universal_extractor',
                'quality_score': self._assess_quality(example),
                'content_type': example.get('type', 'unknown')
            })
        
        return examples
    
    def _detect_structured_qa(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect structured Q&A using regex patterns
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of Q&A examples if structured format detected
        """
        examples = []
        
        for pattern in self.qa_patterns:
            matches = re.finditer(pattern, text, re.DOTALL | re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match.groups()) >= 2:
                    question = self._clean_extracted_text(match.group(1))
                    answer = self._clean_extracted_text(match.group(2))
                    
                    if self._is_valid_qa_pair(question, answer):
                        examples.append({
                            'question': question,
                            'answer': answer,
                            'type': 'structured_qa',
                            'pattern_used': pattern,
                            'confidence': 0.9
                        })
        
        # Remove duplicates and sort by quality
        examples = self._deduplicate_examples(examples)
        return examples
    
    def _detect_dialogue_format(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect dialogue format (speaker-based conversations)
        
        Args:
            text: Text content to analyze
            
        Returns:
            List of dialogue examples if format detected
        """
        examples = []
        lines = text.split('\n')
        
        current_qa = None
        speaker_pattern = re.compile(r'^([A-Z][^:]*?):\s*(.+)$')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = speaker_pattern.match(line)
            if match:
                speaker = match.group(1).strip()
                content = match.group(2).strip()
                
                # Detect question speakers
                if any(keyword in speaker.lower() for keyword in ['question', 'student', 'seeker', 'visitor']):
                    if current_qa:
                        # Save previous Q&A if exists
                        if self._is_valid_qa_pair(current_qa.get('question', ''), current_qa.get('answer', '')):
                            examples.append(current_qa)
                    
                    current_qa = {
                        'question': content,
                        'type': 'dialogue',
                        'questioner': speaker,
                        'confidence': 0.8
                    }
                
                # Detect answer speakers
                elif any(keyword in speaker.lower() for keyword in ['teacher', 'master', 'guru', 'answer']) and current_qa:
                    current_qa.update({
                        'answer': content,
                        'answerer': speaker
                    })
                    
                    if self._is_valid_qa_pair(current_qa.get('question', ''), current_qa.get('answer', '')):
                        examples.append(current_qa)
                    current_qa = None
        
        # Handle last Q&A if exists
        if current_qa and self._is_valid_qa_pair(current_qa.get('question', ''), current_qa.get('answer', '')):
            examples.append(current_qa)
        
        return examples
    
    def _process_monologue_content(self, text: str) -> List[Dict[str, Any]]:
        """
        Process standalone content as monologue with appropriate handling
        
        Args:
            text: Text content to process
            
        Returns:
            List of examples with question="" and answer=paragraph, quality_score=0.3
        """
        examples = []
        
        # Split into meaningful paragraphs
        paragraphs = self._extract_meaningful_paragraphs(text)
        
        for i, paragraph in enumerate(paragraphs):
            if len(paragraph.strip()) > 50:  # Minimum length threshold
                # Generate contextual question for the paragraph
                question = self._generate_contextual_question(paragraph)
                
                examples.append({
                    'question': question,
                    'answer': paragraph,
                    'type': 'monologue',
                    'paragraph_index': i,
                    'confidence': 0.3,  # Lower confidence for monologue content
                    'requires_enhancement': True  # Flag for GPT enhancement
                })
        
        return examples[:20]  # Limit to first 20 paragraphs
    
    def _extract_meaningful_paragraphs(self, text: str) -> List[str]:
        """Extract meaningful paragraphs from text"""
        # Split by double newlines first
        paragraphs = text.split('\n\n')
        
        # If no double newlines, split by single newlines and group
        if len(paragraphs) == 1:
            lines = text.split('\n')
            paragraphs = []
            current_paragraph = []
            
            for line in lines:
                line = line.strip()
                if line:
                    current_paragraph.append(line)
                elif current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
        
        # Clean and filter paragraphs
        meaningful_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if (len(para) > 50 and 
                not self._is_header_or_footer(para) and
                not self._is_page_number(para)):
                meaningful_paragraphs.append(para)
        
        return meaningful_paragraphs
    
    def _generate_contextual_question(self, paragraph: str) -> str:
        """Generate a contextual question for a paragraph"""
        paragraph_lower = paragraph.lower()
        
        # Spiritual content questions
        if any(word in paragraph_lower for word in ['consciousness', 'awareness', 'being']):
            return "What is the nature of consciousness and awareness?"
        elif any(word in paragraph_lower for word in ['meditation', 'practice', 'mindfulness']):
            return "How should one approach spiritual practice?"
        elif any(word in paragraph_lower for word in ['truth', 'reality', 'existence']):
            return "What is the ultimate truth or reality?"
        elif any(word in paragraph_lower for word in ['self', 'ego', 'identity']):
            return "What is the nature of the self?"
        elif any(word in paragraph_lower for word in ['love', 'compassion', 'heart']):
            return "How does love and compassion manifest in spiritual life?"
        elif any(word in paragraph_lower for word in ['suffering', 'pain', 'difficulty']):
            return "How should one understand and work with suffering?"
        elif any(word in paragraph_lower for word in ['enlightenment', 'awakening', 'realization']):
            return "What is the nature of spiritual awakening?"
        elif any(word in paragraph_lower for word in ['god', 'divine', 'sacred']):
            return "How does one relate to the divine or sacred?"
        else:
            return "Can you explain this spiritual teaching?"
    
    def _is_valid_qa_pair(self, question: str, answer: str) -> bool:
        """Check if a Q&A pair is valid"""
        return (len(question.strip()) >= 10 and 
                len(answer.strip()) >= 20 and
                question.strip() != answer.strip())
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text content"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common artifacts
        text = re.sub(r'^[:\-\s]+', '', text)  # Remove leading punctuation
        text = re.sub(r'[:\-\s]+$', '', text)  # Remove trailing punctuation
        
        return text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not isinstance(text, str):
            text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        text = re.sub(r'\n\s*Page \d+\s*\n', '\n', text, flags=re.IGNORECASE)
        
        # Fix common OCR errors
        text = text.replace('—', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        return text.strip()
    
    def _is_header_or_footer(self, text: str) -> bool:
        """Check if text is likely a header or footer"""
        text_lower = text.lower()
        return (len(text) < 100 and 
                any(keyword in text_lower for keyword in 
                    ['chapter', 'page', 'copyright', '©', 'all rights reserved']))
    
    def _is_page_number(self, text: str) -> bool:
        """Check if text is likely a page number"""
        return re.match(r'^\s*\d+\s*$', text) is not None
    
    def _deduplicate_examples(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate examples"""
        seen = set()
        unique_examples = []
        
        for example in examples:
            # Create a signature for the example
            signature = (example.get('question', '').strip().lower()[:50],
                        example.get('answer', '').strip().lower()[:50])
            
            if signature not in seen:
                seen.add(signature)
                unique_examples.append(example)
        
        return unique_examples
    
    def _assess_quality(self, example: Dict[str, Any]) -> float:
        """Assess the quality of an extracted example"""
        score = 0.0
        
        question = example.get('question', '')
        answer = example.get('answer', '')
        content_type = example.get('type', '')
        
        # Base score by content type
        if content_type == 'structured_qa':
            score += 0.4
        elif content_type == 'dialogue':
            score += 0.3
        elif content_type == 'monologue':
            score += 0.1
        
        # Length scoring
        if 20 <= len(question) <= 200:
            score += 0.2
        if 50 <= len(answer) <= 1000:
            score += 0.2
        
        # Content quality
        spiritual_keywords = [
            'consciousness', 'awareness', 'meditation', 'truth', 'reality',
            'being', 'self', 'ego', 'love', 'compassion', 'wisdom',
            'enlightenment', 'awakening', 'mindfulness', 'presence'
        ]
        
        content = (question + ' ' + answer).lower()
        keyword_count = sum(1 for keyword in spiritual_keywords if keyword in content)
        score += min(keyword_count * 0.05, 0.2)
        
        return min(score, 1.0)
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract content from PDF using built-in PDF processing"""
        try:
            # Use built-in PDF processing instead
            return self._extract_pdf_builtin(file_path)
        except Exception as e:
            self.logger.error(f"PDF extraction failed: {e}")
            return ""
    
    def _extract_pdf_builtin(self, file_path: str) -> str:
        """Basic PDF extraction fallback"""
        try:
            import PyPDF2
            text = ""
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            
            return text
            
        except Exception as e:
            logger.error(f"Basic PDF extraction failed: {e}")
            return ""
    
    def _extract_txt(self, file_path: str) -> str:
        """Extract content from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"TXT extraction failed: {e}")
            return ""
    
    def _extract_md(self, file_path: str) -> str:
        """Extract content from Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Markdown extraction failed: {e}")
            return ""
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract content from DOCX file"""
        try:
            from docx import Document
            doc = Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except ImportError:
            logger.error("python-docx not available for DOCX extraction")
            return ""
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            return ""
    
    def _extract_epub(self, file_path: str) -> str:
        """Extract content from EPUB file"""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup
            
            book = epub.read_epub(file_path)
            text = ""
            
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text() + "\n"
            
            return text
            
        except ImportError:
            logger.error("ebooklib not available for EPUB extraction")
            return ""
        except Exception as e:
            logger.error(f"EPUB extraction failed: {e}")
            return ""
    
    def get_extraction_stats(self) -> Dict[str, int]:
        """Get extraction statistics"""
        return self.extraction_stats.copy()
    
    def reset_stats(self):
        """Reset extraction statistics"""
        self.extraction_stats = {
            'total_processed': 0,
            'qa_detected': 0,
            'dialogue_detected': 0,
            'monologue_processed': 0,
            'extraction_errors': 0
        }
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        return self.supported_formats.copy()
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported"""
        return Path(file_path).suffix.lower() in self.supported_formats

