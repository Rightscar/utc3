"""
Smart Content Detection Module
=============================

Provides intelligent detection and processing of different content types.
Implements Core Enhancement 3: Smart Q&A vs Monologue Detection.

Features:
- Structured Q&A pattern detection using regex
- Monologue/paragraph processing for non-Q&A content
- Content type classification and scoring
- Hybrid processing for mixed content
- Quality assessment based on content type
"""

import re
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ContentSegment:
    """Represents a segment of processed content"""
    question: str
    answer: str
    content_type: str  # 'qa', 'monologue', 'passage'
    quality_score: float
    confidence: float
    source_text: str
    metadata: Dict[str, Any]


class SmartContentDetector:
    """Enhanced content detector with Q&A vs monologue intelligence"""
    
    def __init__(self):
        # Q&A detection patterns - Core Enhancement 3 requirement
        self.qa_patterns = [
            # Standard Q&A formats
            r'Q[:\-]\s*(.+?)\nA[:\-]\s*(.+?)(?=\n\n|\nQ[:\-]|$)',
            r'Question[:\-]\s*(.+?)\nAnswer[:\-]\s*(.+?)(?=\n\n|\nQuestion[:\-]|$)',
            r'Q\d+[:\-]\s*(.+?)\nA\d+[:\-]\s*(.+?)(?=\n\n|\nQ\d+[:\-]|$)',
            
            # Interview formats
            r'Interviewer[:\-]\s*(.+?)\n(?:Guest|Interviewee)[:\-]\s*(.+?)(?=\n\n|\nInterviewer[:\-]|$)',
            r'Host[:\-]\s*(.+?)\nGuest[:\-]\s*(.+?)(?=\n\n|\nHost[:\-]|$)',
            
            # Dialogue formats
            r'([A-Z][a-z]+)[:\-]\s*(.+?)\n([A-Z][a-z]+)[:\-]\s*(.+?)(?=\n\n|\n[A-Z][a-z]+[:\-]|$)',
            
            # Spiritual teaching formats
            r'Student[:\-]\s*(.+?)\n(?:Teacher|Master|Guru)[:\-]\s*(.+?)(?=\n\n|\nStudent[:\-]|$)',
            r'Seeker[:\-]\s*(.+?)\n(?:Teacher|Master|Sage)[:\-]\s*(.+?)(?=\n\n|\nSeeker[:\-]|$)',
            
            # Question mark patterns
            r'(.+\?)\s*\n\n(.+?)(?=\n\n.+\?|\n\n[A-Z]|$)',
            
            # Numbered Q&A
            r'(\d+\.\s*.+\?)\s*\n(.+?)(?=\n\d+\.\s*|\n\n|$)',
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.DOTALL | re.IGNORECASE) 
                                for pattern in self.qa_patterns]
        
        # Content type thresholds
        self.qa_threshold = 0.3  # Minimum ratio of Q&A content to classify as dialogue-heavy
        self.min_segment_length = 20  # Minimum characters for a valid segment
        self.max_segment_length = 2000  # Maximum characters for a single segment
        
    def detect_content_type(self, text: str) -> Dict[str, Any]:
        """Detect the primary content type of the text"""
        
        if not text or len(text.strip()) < self.min_segment_length:
            return {
                'primary_type': 'insufficient',
                'qa_ratio': 0.0,
                'confidence': 0.0,
                'total_segments': 0,
                'qa_segments': 0,
                'monologue_segments': 0
            }
        
        # Extract Q&A segments
        qa_segments = self.extract_qa_segments(text)
        
        # Calculate Q&A ratio
        total_text_length = len(text)
        qa_text_length = sum(len(seg.source_text) for seg in qa_segments)
        qa_ratio = qa_text_length / total_text_length if total_text_length > 0 else 0.0
        
        # Determine primary content type
        if qa_ratio >= self.qa_threshold:
            primary_type = 'dialogue'
            confidence = min(qa_ratio * 2, 1.0)  # Higher confidence for higher Q&A ratio
        else:
            primary_type = 'monologue'
            confidence = min((1 - qa_ratio) * 2, 1.0)
            
        # Extract monologue segments for non-Q&A content
        monologue_segments = self.extract_monologue_segments(text, qa_segments)
        
        return {
            'primary_type': primary_type,
            'qa_ratio': qa_ratio,
            'confidence': confidence,
            'total_segments': len(qa_segments) + len(monologue_segments),
            'qa_segments': len(qa_segments),
            'monologue_segments': len(monologue_segments),
            'qa_content': qa_segments,
            'monologue_content': monologue_segments
        }
        
    def extract_qa_segments(self, text: str) -> List[ContentSegment]:
        """Extract Q&A segments using regex patterns - Core Enhancement 3 requirement"""
        
        qa_segments = []
        processed_positions = set()
        
        for pattern in self.compiled_patterns:
            matches = pattern.finditer(text)
            
            for match in matches:
                start, end = match.span()
                
                # Skip if this text has already been processed
                if any(pos in processed_positions for pos in range(start, end)):
                    continue
                    
                # Mark positions as processed
                processed_positions.update(range(start, end))
                
                groups = match.groups()
                
                if len(groups) >= 2:
                    question = groups[0].strip()
                    answer = groups[1].strip()
                    
                    # Validate segment quality
                    if (len(question) >= self.min_segment_length and 
                        len(answer) >= self.min_segment_length and
                        len(question) <= self.max_segment_length and
                        len(answer) <= self.max_segment_length):
                        
                        # Calculate quality score
                        quality_score = self._calculate_qa_quality(question, answer)
                        
                        segment = ContentSegment(
                            question=question,
                            answer=answer,
                            content_type='qa',
                            quality_score=quality_score,
                            confidence=0.8,  # High confidence for regex matches
                            source_text=match.group(0),
                            metadata={
                                'pattern_used': pattern.pattern,
                                'position': (start, end),
                                'question_length': len(question),
                                'answer_length': len(answer)
                            }
                        )
                        
                        qa_segments.append(segment)
        
        # Sort by position in text
        qa_segments.sort(key=lambda x: x.metadata['position'][0])
        
        logger.info(f"Extracted {len(qa_segments)} Q&A segments")
        return qa_segments
        
    def extract_monologue_segments(self, text: str, qa_segments: List[ContentSegment]) -> List[ContentSegment]:
        """Extract monologue segments from non-Q&A content - Core Enhancement 3 requirement"""
        
        # Get positions of Q&A segments to avoid overlap
        qa_positions = set()
        for segment in qa_segments:
            start, end = segment.metadata['position']
            qa_positions.update(range(start, end))
        
        # Split text into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        monologue_segments = []
        current_position = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            
            if len(paragraph) < self.min_segment_length:
                current_position += len(paragraph) + 2  # +2 for paragraph breaks
                continue
                
            # Find paragraph position in original text
            para_start = text.find(paragraph, current_position)
            para_end = para_start + len(paragraph)
            
            # Skip if this paragraph overlaps with Q&A content
            if any(pos in qa_positions for pos in range(para_start, para_end)):
                current_position = para_end + 2
                continue
            
            # Process paragraph as standalone entry - Core Enhancement 3 requirement
            if len(paragraph) <= self.max_segment_length:
                # Use paragraph as both question and answer for monologue content
                quality_score = self._calculate_monologue_quality(paragraph)
                
                segment = ContentSegment(
                    question=paragraph,  # Paragraph as question
                    answer="",  # Empty answer for monologue - Core Enhancement 3 requirement
                    content_type='monologue',
                    quality_score=quality_score,
                    confidence=0.6,  # Lower confidence for monologue extraction
                    source_text=paragraph,
                    metadata={
                        'position': (para_start, para_end),
                        'paragraph_length': len(paragraph),
                        'word_count': len(paragraph.split()),
                        'sentence_count': len(re.split(r'[.!?]+', paragraph))
                    }
                )
                
                monologue_segments.append(segment)
            else:
                # Split long paragraphs into smaller segments
                sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                current_segment = ""
                
                for sentence in sentences:
                    if len(current_segment + sentence) <= self.max_segment_length:
                        current_segment += sentence + " "
                    else:
                        if len(current_segment.strip()) >= self.min_segment_length:
                            quality_score = self._calculate_monologue_quality(current_segment.strip())
                            
                            segment = ContentSegment(
                                question=current_segment.strip(),
                                answer="",
                                content_type='passage',
                                quality_score=quality_score,
                                confidence=0.5,
                                source_text=current_segment.strip(),
                                metadata={
                                    'position': (para_start, para_start + len(current_segment)),
                                    'segment_length': len(current_segment),
                                    'word_count': len(current_segment.split()),
                                    'is_split_paragraph': True
                                }
                            )
                            
                            monologue_segments.append(segment)
                        
                        current_segment = sentence + " "
                
                # Add remaining segment
                if len(current_segment.strip()) >= self.min_segment_length:
                    quality_score = self._calculate_monologue_quality(current_segment.strip())
                    
                    segment = ContentSegment(
                        question=current_segment.strip(),
                        answer="",
                        content_type='passage',
                        quality_score=quality_score,
                        confidence=0.5,
                        source_text=current_segment.strip(),
                        metadata={
                            'position': (para_start, para_start + len(current_segment)),
                            'segment_length': len(current_segment),
                            'word_count': len(current_segment.split()),
                            'is_split_paragraph': True
                        }
                    )
                    
                    monologue_segments.append(segment)
            
            current_position = para_end + 2
        
        logger.info(f"Extracted {len(monologue_segments)} monologue segments")
        return monologue_segments
        
    def _calculate_qa_quality(self, question: str, answer: str) -> float:
        """Calculate quality score for Q&A segments"""
        
        score = 0.0
        
        # Length factors
        q_len = len(question.split())
        a_len = len(answer.split())
        
        # Optimal length ranges
        if 5 <= q_len <= 50:
            score += 0.2
        if 10 <= a_len <= 200:
            score += 0.3
        
        # Question quality indicators
        if question.strip().endswith('?'):
            score += 0.1
        if any(word in question.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            score += 0.1
        
        # Answer quality indicators
        if len(answer) > len(question):  # Answers should generally be longer
            score += 0.1
        if any(word in answer.lower() for word in ['because', 'therefore', 'however', 'thus']):
            score += 0.1
        
        # Spiritual content indicators
        spiritual_keywords = ['consciousness', 'awareness', 'meditation', 'spiritual', 'enlightenment', 
                            'wisdom', 'truth', 'being', 'presence', 'mindfulness', 'soul', 'divine']
        
        spiritual_count = sum(1 for word in spiritual_keywords 
                            if word in (question + ' ' + answer).lower())
        score += min(spiritual_count * 0.05, 0.2)
        
        return min(score, 1.0)
        
    def _calculate_monologue_quality(self, text: str) -> float:
        """Calculate quality score for monologue segments"""
        
        # Base quality for monologue content - Core Enhancement 3 requirement
        score = 0.3  # Lower base score as specified
        
        # Length factors
        word_count = len(text.split())
        
        # Optimal length range for monologue
        if 20 <= word_count <= 150:
            score += 0.2
        elif 10 <= word_count <= 300:
            score += 0.1
        
        # Content depth indicators
        if len(re.split(r'[.!?]+', text)) >= 2:  # Multiple sentences
            score += 0.1
        
        # Spiritual content indicators
        spiritual_keywords = ['consciousness', 'awareness', 'meditation', 'spiritual', 'enlightenment', 
                            'wisdom', 'truth', 'being', 'presence', 'mindfulness', 'soul', 'divine',
                            'peace', 'love', 'compassion', 'understanding', 'realization']
        
        spiritual_count = sum(1 for word in spiritual_keywords if word in text.lower())
        score += min(spiritual_count * 0.05, 0.2)
        
        # Complexity indicators
        if any(word in text.lower() for word in ['therefore', 'however', 'moreover', 'furthermore']):
            score += 0.05
        
        return min(score, 1.0)
        
    def process_content(self, text: str) -> Dict[str, Any]:
        """Process content with smart detection - Main interface method"""
        
        logger.info("Starting smart content detection and processing")
        
        # Detect content type
        detection_result = self.detect_content_type(text)
        
        # Combine all segments
        all_segments = []
        if 'qa_content' in detection_result:
            all_segments.extend(detection_result['qa_content'])
        if 'monologue_content' in detection_result:
            all_segments.extend(detection_result['monologue_content'])
        
        # Convert segments to standard format
        processed_content = []
        for segment in all_segments:
            content_item = {
                'question': segment.question,
                'answer': segment.answer,
                'content_type': segment.content_type,
                'quality_score': segment.quality_score,
                'confidence': segment.confidence,
                'source_text': segment.source_text,
                'metadata': segment.metadata,
                'enhanced': False,
                'include': True  # Default to include
            }
            processed_content.append(content_item)
        
        # Sort by quality score (highest first)
        processed_content.sort(key=lambda x: x['quality_score'], reverse=True)
        
        result = {
            'content': processed_content,
            'detection_summary': detection_result,
            'total_segments': len(processed_content),
            'qa_segments': len([c for c in processed_content if c['content_type'] == 'qa']),
            'monologue_segments': len([c for c in processed_content if c['content_type'] in ['monologue', 'passage']]),
            'average_quality': sum(c['quality_score'] for c in processed_content) / len(processed_content) if processed_content else 0.0
        }
        
        logger.info(f"Smart content processing complete: {result['total_segments']} segments, "
                   f"{result['qa_segments']} Q&A, {result['monologue_segments']} monologue")
        
        return result
        
    def render_detection_summary(self, detection_result: Dict[str, Any]) -> None:
        """Render content detection summary in Streamlit"""
        
        st.markdown("### 🔍 **Smart Content Detection Results**")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Content Type", detection_result['primary_type'].title())
            
        with col2:
            st.metric("Q&A Ratio", f"{detection_result['qa_ratio']:.1%}")
            
        with col3:
            st.metric("Confidence", f"{detection_result['confidence']:.1%}")
            
        with col4:
            st.metric("Total Segments", detection_result['total_segments'])
        
        # Detailed breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🗣️ Q&A Segments", detection_result['qa_segments'])
            
        with col2:
            st.metric("📝 Monologue Segments", detection_result['monologue_segments'])
        
        # Content type explanation
        if detection_result['primary_type'] == 'dialogue':
            st.success("✅ **Dialogue-Heavy Content**: This content contains structured Q&A patterns and will be processed primarily as conversations.")
        elif detection_result['primary_type'] == 'monologue':
            st.info("📝 **Monologue Content**: This content will be processed as standalone passages and converted to training examples.")
        else:
            st.warning("⚠️ **Insufficient Content**: Not enough content for reliable processing.")
            
    def get_processing_recommendations(self, detection_result: Dict[str, Any]) -> List[str]:
        """Get processing recommendations based on content type"""
        
        recommendations = []
        
        if detection_result['primary_type'] == 'dialogue':
            recommendations.append("✅ Use Q&A enhancement mode for best results")
            recommendations.append("🎯 Focus on dialogue-specific spiritual tones")
            recommendations.append("📊 Expect high-quality training data output")
            
        elif detection_result['primary_type'] == 'monologue':
            recommendations.append("📝 Content will be processed as standalone passages")
            recommendations.append("🔄 Consider generating synthetic Q&A from passages")
            recommendations.append("⚠️ Quality scores may be lower (0.3 base)")
            recommendations.append("🎨 Use narrative-focused enhancement tones")
            
        if detection_result['qa_ratio'] > 0 and detection_result['qa_ratio'] < self.qa_threshold:
            recommendations.append("🔀 Mixed content detected - hybrid processing will be used")
            
        if detection_result['total_segments'] < 10:
            recommendations.append("📈 Consider uploading additional content for better training data")
            
        return recommendations

