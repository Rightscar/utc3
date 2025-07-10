"""
Enhanced Quality Control Module
==============================

Provides advanced quality control capabilities for the Enhanced Universal AI Training Data Creator.
Includes semantic similarity scoring, hallucination detection, and comprehensive quality metrics.

Features:
- Semantic similarity scoring using sentence transformers
- Hallucination detection and flagging
- Multi-dimensional quality assessment
- Automatic quality threshold handling
- Quality trend analysis and reporting
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import re
import json

logger = logging.getLogger(__name__)


class EnhancedQualityControl:
    """Enhanced quality control system with semantic analysis"""
    
    def __init__(self):
        self.quality_history = []
        self.threshold_config = {
            'semantic_similarity_min': 0.75,
            'length_ratio_max': 1.8,
            'length_ratio_min': 0.5,
            'hallucination_threshold': 0.3,
            'overall_quality_min': 0.7
        }
        self.sentence_transformer = None
        self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        """Load sentence transformer model for semantic similarity"""
        try:
            from sentence_transformers import SentenceTransformer
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except ImportError:
            logger.warning("sentence-transformers not available, using fallback similarity")
            self.sentence_transformer = None
        except Exception as e:
            logger.error(f"Error loading sentence transformer: {e}")
            self.sentence_transformer = None
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        if self.sentence_transformer is not None:
            try:
                # Use sentence transformer for accurate similarity
                embeddings = self.sentence_transformer.encode([text1, text2])
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                return float(similarity)
            except Exception as e:
                logger.warning(f"Error computing semantic similarity: {e}")
                return self._fallback_similarity(text1, text2)
        else:
            return self._fallback_similarity(text1, text2)
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback similarity computation using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def detect_hallucinations(self, original: str, enhanced: str) -> Dict[str, Any]:
        """Detect potential hallucinations in enhanced content"""
        hallucination_indicators = {
            'factual_claims': [],
            'new_entities': [],
            'contradictions': [],
            'unsupported_details': [],
            'hallucination_score': 0.0
        }
        
        # Extract entities and claims
        original_entities = self._extract_entities(original)
        enhanced_entities = self._extract_entities(enhanced)
        
        # Find new entities not in original
        new_entities = enhanced_entities - original_entities
        hallucination_indicators['new_entities'] = list(new_entities)
        
        # Check for factual claims
        factual_patterns = [
            r'\b(studies show|research indicates|scientists found|proven that)\b',
            r'\b(\d+%|\d+ percent|statistics show)\b',
            r'\b(according to|based on research|evidence suggests)\b'
        ]
        
        for pattern in factual_patterns:
            matches = re.findall(pattern, enhanced, re.IGNORECASE)
            if matches and not re.search(pattern, original, re.IGNORECASE):
                hallucination_indicators['factual_claims'].extend(matches)
        
        # Check for contradictions
        contradictions = self._detect_contradictions(original, enhanced)
        hallucination_indicators['contradictions'] = contradictions
        
        # Check for unsupported details
        unsupported = self._detect_unsupported_details(original, enhanced)
        hallucination_indicators['unsupported_details'] = unsupported
        
        # Calculate hallucination score
        score_components = [
            len(new_entities) * 0.1,
            len(hallucination_indicators['factual_claims']) * 0.2,
            len(contradictions) * 0.3,
            len(unsupported) * 0.1
        ]
        
        hallucination_indicators['hallucination_score'] = min(sum(score_components), 1.0)
        
        return hallucination_indicators
    
    def _extract_entities(self, text: str) -> set:
        """Extract named entities and key terms from text"""
        # Simple entity extraction (could be enhanced with NLP libraries)
        entities = set()
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
        entities.update(capitalized_words)
        
        # Extract quoted terms
        quoted_terms = re.findall(r'"([^"]+)"', text)
        entities.update(quoted_terms)
        
        # Extract numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        entities.update(numbers)
        
        return entities
    
    def _detect_contradictions(self, original: str, enhanced: str) -> List[str]:
        """Detect potential contradictions between original and enhanced text"""
        contradictions = []
        
        # Simple contradiction detection patterns
        contradiction_patterns = [
            (r'\bnot\s+(\w+)', r'\b\1\b'),  # "not X" vs "X"
            (r'\bno\s+(\w+)', r'\b\1\b'),   # "no X" vs "X"
            (r'\bfalse', r'\btrue'),        # "false" vs "true"
            (r'\bimpossible', r'\bpossible'), # "impossible" vs "possible"
        ]
        
        for neg_pattern, pos_pattern in contradiction_patterns:
            neg_matches = re.findall(neg_pattern, original, re.IGNORECASE)
            for match in neg_matches:
                if re.search(pos_pattern.replace(r'\1', match), enhanced, re.IGNORECASE):
                    contradictions.append(f"Contradiction: '{match}' negated in original but affirmed in enhanced")
        
        return contradictions
    
    def _detect_unsupported_details(self, original: str, enhanced: str) -> List[str]:
        """Detect details in enhanced text not supported by original"""
        unsupported = []
        
        # Check for specific details that weren't in original
        enhanced_sentences = enhanced.split('.')
        original_lower = original.lower()
        
        for sentence in enhanced_sentences:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Only check substantial sentences
                # Check if sentence contains information not in original
                sentence_words = set(sentence.lower().split())
                original_words = set(original_lower.split())
                
                new_words = sentence_words - original_words
                if len(new_words) > len(sentence_words) * 0.7:  # >70% new words
                    unsupported.append(sentence[:100] + "..." if len(sentence) > 100 else sentence)
        
        return unsupported[:5]  # Limit to 5 examples
    
    def comprehensive_quality_assessment(self, original: str, enhanced: str, 
                                       tone: str = "") -> Dict[str, Any]:
        """Perform comprehensive quality assessment"""
        assessment = {
            'timestamp': datetime.now().isoformat(),
            'tone': tone,
            'metrics': {},
            'flags': [],
            'overall_score': 0.0,
            'passed_threshold': False,
            'recommendations': []
        }
        
        # Basic metrics
        length_ratio = len(enhanced) / len(original) if original else 0
        assessment['metrics']['length_ratio'] = length_ratio
        
        # Semantic similarity
        semantic_similarity = self.compute_semantic_similarity(original, enhanced)
        assessment['metrics']['semantic_similarity'] = semantic_similarity
        
        # Hallucination detection
        hallucination_data = self.detect_hallucinations(original, enhanced)
        assessment['metrics']['hallucination_score'] = hallucination_data['hallucination_score']
        assessment['hallucination_details'] = hallucination_data
        
        # Content quality metrics
        content_quality = self._assess_content_quality(enhanced)
        assessment['metrics'].update(content_quality)
        
        # Tone consistency (if tone specified)
        if tone:
            tone_consistency = self._assess_tone_consistency(enhanced, tone)
            assessment['metrics']['tone_consistency'] = tone_consistency
        
        # Generate flags based on thresholds
        flags = []
        
        if semantic_similarity < self.threshold_config['semantic_similarity_min']:
            flags.append(f'low_semantic_similarity_{semantic_similarity:.3f}')
        
        if length_ratio > self.threshold_config['length_ratio_max']:
            flags.append(f'excessive_expansion_{length_ratio:.2f}')
        elif length_ratio < self.threshold_config['length_ratio_min']:
            flags.append(f'excessive_reduction_{length_ratio:.2f}')
        
        if hallucination_data['hallucination_score'] > self.threshold_config['hallucination_threshold']:
            flags.append(f'potential_hallucination_{hallucination_data["hallucination_score"]:.3f}')
        
        if content_quality['readability_score'] < 0.5:
            flags.append('low_readability')
        
        if content_quality['coherence_score'] < 0.6:
            flags.append('low_coherence')
        
        assessment['flags'] = flags
        
        # Calculate overall score
        score_components = [
            semantic_similarity * 0.3,
            (1 - hallucination_data['hallucination_score']) * 0.2,
            content_quality['readability_score'] * 0.2,
            content_quality['coherence_score'] * 0.2,
            self._length_ratio_score(length_ratio) * 0.1
        ]
        
        if tone:
            score_components.append(assessment['metrics']['tone_consistency'] * 0.1)
            # Normalize weights
            score_components = [s * 0.9 for s in score_components[:-1]] + [score_components[-1]]
        
        assessment['overall_score'] = sum(score_components)
        
        # Determine if passed threshold
        assessment['passed_threshold'] = (
            assessment['overall_score'] >= self.threshold_config['overall_quality_min'] and
            len(flags) == 0
        )
        
        # Generate recommendations
        assessment['recommendations'] = self._generate_recommendations(assessment)
        
        # Store in history
        self.quality_history.append(assessment)
        
        return assessment
    
    def _assess_content_quality(self, content: str) -> Dict[str, float]:
        """Assess content quality metrics"""
        if not content:
            return {'readability_score': 0.0, 'coherence_score': 0.0, 'completeness_score': 0.0}
        
        sentences = content.split('.')
        words = content.split()
        
        # Readability score (based on sentence length and word complexity)
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Optimal sentence length: 15-20 words
        sentence_length_score = 1.0 - abs(avg_sentence_length - 17.5) / 17.5
        sentence_length_score = max(0, min(1, sentence_length_score))
        
        # Optimal word length: 4-6 characters
        word_length_score = 1.0 - abs(avg_word_length - 5) / 5
        word_length_score = max(0, min(1, word_length_score))
        
        readability_score = (sentence_length_score + word_length_score) / 2
        
        # Coherence score (based on sentence connectivity)
        coherence_score = self._assess_coherence(sentences)
        
        # Completeness score (based on content structure)
        completeness_score = self._assess_completeness(content)
        
        return {
            'readability_score': readability_score,
            'coherence_score': coherence_score,
            'completeness_score': completeness_score
        }
    
    def _assess_coherence(self, sentences: List[str]) -> float:
        """Assess coherence between sentences"""
        if len(sentences) < 2:
            return 1.0
        
        coherence_indicators = 0
        total_transitions = len(sentences) - 1
        
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'similarly', 'in contrast', 'for example',
            'specifically', 'in particular', 'as a result', 'on the other hand'
        ]
        
        for i in range(len(sentences) - 1):
            current_sentence = sentences[i].lower()
            next_sentence = sentences[i + 1].lower()
            
            # Check for transition words
            if any(word in next_sentence for word in transition_words):
                coherence_indicators += 1
            
            # Check for word overlap between adjacent sentences
            current_words = set(current_sentence.split())
            next_words = set(next_sentence.split())
            overlap = len(current_words & next_words)
            
            if overlap >= 2:  # At least 2 words in common
                coherence_indicators += 0.5
        
        return min(coherence_indicators / total_transitions, 1.0) if total_transitions > 0 else 1.0
    
    def _assess_completeness(self, content: str) -> float:
        """Assess content completeness"""
        completeness_indicators = 0
        
        # Check for introduction/conclusion patterns
        if any(word in content.lower() for word in ['introduction', 'begin', 'start', 'first']):
            completeness_indicators += 0.3
        
        if any(word in content.lower() for word in ['conclusion', 'summary', 'finally', 'in conclusion']):
            completeness_indicators += 0.3
        
        # Check for examples or explanations
        if any(word in content.lower() for word in ['example', 'for instance', 'such as', 'specifically']):
            completeness_indicators += 0.2
        
        # Check for balanced structure (multiple paragraphs/sections)
        paragraphs = content.split('\n\n')
        if len(paragraphs) >= 2:
            completeness_indicators += 0.2
        
        return min(completeness_indicators, 1.0)
    
    def _assess_tone_consistency(self, content: str, target_tone: str) -> float:
        """Assess consistency with target tone"""
        tone_keywords = {
            'zen_buddhism': ['mindfulness', 'awareness', 'present', 'meditation', 'compassion'],
            'advaita_vedanta': ['consciousness', 'awareness', 'self', 'reality', 'truth'],
            'christian_mysticism': ['divine', 'sacred', 'prayer', 'contemplation', 'grace'],
            'sufi_mysticism': ['heart', 'love', 'divine', 'soul', 'spiritual'],
            'mindfulness_meditation': ['present', 'awareness', 'attention', 'mindful', 'breath'],
            'universal_wisdom': ['wisdom', 'understanding', 'truth', 'insight', 'knowledge']
        }
        
        if target_tone not in tone_keywords:
            return 0.5  # Neutral score for unknown tones
        
        keywords = tone_keywords[target_tone]
        content_lower = content.lower()
        
        keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
        tone_score = keyword_count / len(keywords)
        
        return min(tone_score, 1.0)
    
    def _length_ratio_score(self, ratio: float) -> float:
        """Score length ratio (optimal around 1.2-1.5)"""
        optimal_ratio = 1.35
        deviation = abs(ratio - optimal_ratio)
        score = 1.0 - (deviation / optimal_ratio)
        return max(0, min(1, score))
    
    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on assessment"""
        recommendations = []
        
        metrics = assessment['metrics']
        flags = assessment['flags']
        
        if metrics['semantic_similarity'] < 0.7:
            recommendations.append("Consider staying closer to the original content's meaning")
        
        if metrics['hallucination_score'] > 0.3:
            recommendations.append("Review for potential hallucinations or unsupported claims")
        
        if any('expansion' in flag for flag in flags):
            recommendations.append("Content may be too verbose; consider condensing")
        
        if any('reduction' in flag for flag in flags):
            recommendations.append("Content may be too brief; consider expanding key points")
        
        if metrics.get('readability_score', 1) < 0.6:
            recommendations.append("Improve readability with shorter sentences and simpler words")
        
        if metrics.get('coherence_score', 1) < 0.6:
            recommendations.append("Add transition words to improve flow between sentences")
        
        if not assessment['passed_threshold']:
            recommendations.append("Content requires manual review before export")
        
        return recommendations
    
    def get_quality_trends(self) -> Dict[str, Any]:
        """Analyze quality trends over time"""
        if not self.quality_history:
            return {'message': 'No quality data available yet'}
        
        recent_assessments = self.quality_history[-10:]  # Last 10 assessments
        
        avg_overall_score = sum(a['overall_score'] for a in recent_assessments) / len(recent_assessments)
        avg_semantic_similarity = sum(a['metrics']['semantic_similarity'] for a in recent_assessments) / len(recent_assessments)
        avg_hallucination_score = sum(a['metrics']['hallucination_score'] for a in recent_assessments) / len(recent_assessments)
        
        pass_rate = sum(1 for a in recent_assessments if a['passed_threshold']) / len(recent_assessments)
        
        common_flags = {}
        for assessment in recent_assessments:
            for flag in assessment['flags']:
                common_flags[flag] = common_flags.get(flag, 0) + 1
        
        return {
            'total_assessments': len(self.quality_history),
            'recent_assessments': len(recent_assessments),
            'avg_overall_score': avg_overall_score,
            'avg_semantic_similarity': avg_semantic_similarity,
            'avg_hallucination_score': avg_hallucination_score,
            'pass_rate': pass_rate,
            'common_flags': sorted(common_flags.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def should_route_to_manual_review(self, assessment: Dict[str, Any]) -> bool:
        """Determine if content should be routed to manual review"""
        return not assessment['passed_threshold']


# Global enhanced quality control instance
enhanced_quality_control = EnhancedQualityControl()

