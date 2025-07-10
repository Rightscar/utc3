#!/usr/bin/env python3
"""
Quality Monitor Module
Real-time content quality analysis and monitoring
"""

import re
import statistics
from typing import Dict, List, Any, Tuple, Optional
import streamlit as st
import logging
from datetime import datetime

try:
    import spacy
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

class QualityMonitor:
    def __init__(self):
        self.quality_thresholds = {
            'min_chunk_size': 20,  # minimum words
            'max_chunk_size': 1000,  # maximum words
            'min_coherence': 0.5,
            'min_dialogue_potential': 0.3,
            'min_readability': 30,  # Flesch reading ease
            'max_readability': 90,
            'min_sentence_variety': 0.4,
            'min_vocabulary_diversity': 0.3
        }
        
        self.quality_weights = {
            'size': 0.15,
            'coherence': 0.25,
            'dialogue_potential': 0.20,
            'readability': 0.15,
            'sentence_variety': 0.10,
            'vocabulary_diversity': 0.15
        }
        
        # Initialize spaCy if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.warning("⚠️ spaCy model not found. Some quality features will be limited.")
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_content_quality(self, content: str) -> Dict[str, Any]:
        """Comprehensive content quality analysis"""
        if not content or not content.strip():
            return self.get_empty_analysis()
        
        try:
            analysis = {
                'content_length': len(content),
                'word_count': len(content.split()),
                'sentence_count': self.count_sentences(content),
                'paragraph_count': len([p for p in content.split('\n\n') if p.strip()]),
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'readability_score': 0,
                'dialogue_potential': 0,
                'coherence_score': 0,
                'sentence_variety': 0,
                'vocabulary_diversity': 0,
                'quality_score': 0,
                'quality_grade': 'Unknown',
                'issues': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Basic metrics
            words = content.split()
            sentences = self.split_sentences(content)
            
            if words:
                analysis['avg_word_length'] = sum(len(word) for word in words) / len(words)
            
            if sentences:
                analysis['avg_sentence_length'] = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
            
            # Advanced analysis
            analysis['readability_score'] = self.calculate_readability(content)
            analysis['dialogue_potential'] = self.assess_dialogue_potential(content)
            analysis['coherence_score'] = self.assess_coherence(content)
            analysis['sentence_variety'] = self.assess_sentence_variety(sentences)
            analysis['vocabulary_diversity'] = self.calculate_vocabulary_diversity(words)
            
            # Overall quality score
            analysis['quality_score'] = self.calculate_overall_quality(analysis)
            analysis['quality_grade'] = self.get_quality_grade(analysis['quality_score'])
            
            # Identify issues and recommendations
            analysis['issues'], analysis['warnings'], analysis['recommendations'] = self.identify_quality_issues(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Content quality analysis failed: {str(e)}")
            return self.get_error_analysis(str(e))
    
    def get_empty_analysis(self) -> Dict[str, Any]:
        """Return analysis for empty content"""
        return {
            'content_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'quality_score': 0,
            'quality_grade': 'Empty',
            'issues': ['Content is empty'],
            'warnings': [],
            'recommendations': ['Add content to analyze']
        }
    
    def get_error_analysis(self, error_msg: str) -> Dict[str, Any]:
        """Return analysis for error cases"""
        return {
            'content_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'quality_score': 0,
            'quality_grade': 'Error',
            'issues': [f'Analysis failed: {error_msg}'],
            'warnings': [],
            'recommendations': ['Try analyzing the content again']
        }
    
    def count_sentences(self, text: str) -> int:
        """Count sentences in text"""
        if self.nlp:
            doc = self.nlp(text)
            return len(list(doc.sents))
        else:
            # Fallback method
            sentences = re.split(r'[.!?]+', text)
            return len([s for s in sentences if s.strip()])
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            # Fallback method
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def calculate_readability(self, text: str) -> float:
        """Calculate readability score"""
        try:
            if SPACY_AVAILABLE:
                return flesch_reading_ease(text)
            else:
                # Simple fallback calculation
                words = text.split()
                sentences = self.split_sentences(text)
                
                if not words or not sentences:
                    return 0
                
                avg_sentence_length = len(words) / len(sentences)
                avg_syllables = sum(self.count_syllables(word) for word in words) / len(words)
                
                # Simplified Flesch formula
                score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
                return max(0, min(100, score))
        except Exception:
            return 50  # Default middle score
    
    def count_syllables(self, word: str) -> int:
        """Simple syllable counting"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_was_vowel:
                syllable_count += 1
            prev_was_vowel = is_vowel
        
        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1
        
        return max(1, syllable_count)
    
    def assess_dialogue_potential(self, text: str) -> float:
        """Assess potential for dialogue generation"""
        dialogue_indicators = [
            r'"[^"]*"',  # Quoted speech
            r"'[^']*'",  # Single quoted speech
            r'\b(said|asked|replied|answered|explained|stated|declared)\b',  # Speech verbs
            r'\b(he|she|they|I|you)\s+(said|asked|replied)\b',  # Dialogue patterns
            r'[?!]',  # Questions and exclamations
            r'\b(what|how|why|when|where|who)\b',  # Question words
            r'\b(yes|no|okay|alright|sure)\b',  # Response words
        ]
        
        total_matches = 0
        for pattern in dialogue_indicators:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            total_matches += matches
        
        # Normalize by text length
        words = len(text.split())
        if words == 0:
            return 0
        
        dialogue_density = total_matches / words
        return min(1.0, dialogue_density * 10)  # Scale to 0-1
    
    def assess_coherence(self, text: str) -> float:
        """Assess text coherence"""
        if self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [sent.text for sent in doc.sents]
                
                if len(sentences) < 2:
                    return 0.8  # Single sentence is coherent by default
                
                # Check for transition words and coherence indicators
                transition_words = [
                    'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                    'consequently', 'meanwhile', 'nevertheless', 'thus', 'hence',
                    'first', 'second', 'finally', 'next', 'then', 'also', 'because'
                ]
                
                transition_count = 0
                for word in transition_words:
                    if word in text.lower():
                        transition_count += 1
                
                # Simple coherence score based on transitions and sentence structure
                coherence = min(1.0, transition_count / len(sentences) + 0.5)
                return coherence
                
            except Exception:
                pass
        
        # Fallback coherence assessment
        sentences = self.split_sentences(text)
        if len(sentences) < 2:
            return 0.8
        
        # Check for basic coherence indicators
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_variance = statistics.variance([len(s.split()) for s in sentences]) if len(sentences) > 1 else 0
        
        # Coherence based on sentence length consistency
        coherence = max(0.3, 1.0 - (length_variance / (avg_sentence_length ** 2)))
        return min(1.0, coherence)
    
    def assess_sentence_variety(self, sentences: List[str]) -> float:
        """Assess sentence structure variety"""
        if not sentences:
            return 0
        
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        
        if len(sentence_lengths) < 2:
            return 0.5
        
        # Calculate coefficient of variation
        mean_length = statistics.mean(sentence_lengths)
        if mean_length == 0:
            return 0
        
        std_dev = statistics.stdev(sentence_lengths)
        variety_score = std_dev / mean_length
        
        return min(1.0, variety_score)
    
    def calculate_vocabulary_diversity(self, words: List[str]) -> float:
        """Calculate vocabulary diversity (Type-Token Ratio)"""
        if not words:
            return 0
        
        # Clean words
        clean_words = [word.lower().strip('.,!?";:()[]{}') for word in words if word.isalpha()]
        
        if not clean_words:
            return 0
        
        unique_words = set(clean_words)
        diversity = len(unique_words) / len(clean_words)
        
        return diversity
    
    def calculate_overall_quality(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        scores = {}
        
        # Size score
        word_count = analysis['word_count']
        if word_count < self.quality_thresholds['min_chunk_size']:
            scores['size'] = 0.2
        elif word_count > self.quality_thresholds['max_chunk_size']:
            scores['size'] = 0.7
        else:
            # Optimal range
            optimal_min = self.quality_thresholds['min_chunk_size']
            optimal_max = 500  # Sweet spot
            if word_count <= optimal_max:
                scores['size'] = 0.8 + 0.2 * ((word_count - optimal_min) / (optimal_max - optimal_min))
            else:
                scores['size'] = 0.8
        
        # Other scores (already normalized 0-1)
        scores['coherence'] = analysis['coherence_score']
        scores['dialogue_potential'] = analysis['dialogue_potential']
        scores['sentence_variety'] = analysis['sentence_variety']
        scores['vocabulary_diversity'] = analysis['vocabulary_diversity']
        
        # Readability score (convert to 0-1)
        readability = analysis['readability_score']
        if readability < self.quality_thresholds['min_readability']:
            scores['readability'] = 0.3
        elif readability > self.quality_thresholds['max_readability']:
            scores['readability'] = 0.7
        else:
            scores['readability'] = 0.8
        
        # Weighted average
        total_score = sum(scores[key] * self.quality_weights[key] for key in scores)
        return round(total_score, 3)
    
    def get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        elif score >= 0.5:
            return "Poor"
        else:
            return "Very Poor"
    
    def identify_quality_issues(self, analysis: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """Identify quality issues, warnings, and recommendations"""
        issues = []
        warnings = []
        recommendations = []
        
        word_count = analysis['word_count']
        
        # Size issues
        if word_count < self.quality_thresholds['min_chunk_size']:
            issues.append(f"Content too short ({word_count} words)")
            recommendations.append("Add more content or combine with other chunks")
        elif word_count > self.quality_thresholds['max_chunk_size']:
            warnings.append(f"Content very long ({word_count} words)")
            recommendations.append("Consider splitting into smaller chunks")
        
        # Coherence issues
        if analysis['coherence_score'] < self.quality_thresholds['min_coherence']:
            issues.append("Low coherence - content may be fragmented")
            recommendations.append("Review content structure and add transitions")
        
        # Dialogue potential
        if analysis['dialogue_potential'] < self.quality_thresholds['min_dialogue_potential']:
            warnings.append("Low dialogue potential")
            recommendations.append("Consider using narrative-to-dialogue enhancement")
        
        # Readability issues
        readability = analysis['readability_score']
        if readability < self.quality_thresholds['min_readability']:
            warnings.append("Text may be too complex")
            recommendations.append("Simplify sentence structure and vocabulary")
        elif readability > self.quality_thresholds['max_readability']:
            warnings.append("Text may be too simple")
            recommendations.append("Add more varied sentence structures")
        
        # Sentence variety
        if analysis['sentence_variety'] < self.quality_thresholds['min_sentence_variety']:
            warnings.append("Low sentence variety")
            recommendations.append("Vary sentence lengths and structures")
        
        # Vocabulary diversity
        if analysis['vocabulary_diversity'] < self.quality_thresholds['min_vocabulary_diversity']:
            warnings.append("Limited vocabulary diversity")
            recommendations.append("Use more varied vocabulary")
        
        return issues, warnings, recommendations
    
    def analyze_chunk_batch(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze quality of multiple chunks"""
        if not chunks:
            return {'total_chunks': 0, 'analysis_complete': False}
        
        batch_analysis = {
            'total_chunks': len(chunks),
            'analyzed_chunks': 0,
            'quality_scores': [],
            'quality_distribution': {},
            'avg_quality': 0,
            'issues_count': 0,
            'warnings_count': 0,
            'recommendations': [],
            'analysis_complete': False
        }
        
        try:
            for i, chunk in enumerate(chunks):
                content = chunk.get('content', '')
                if content:
                    analysis = self.analyze_content_quality(content)
                    
                    # Update chunk with analysis
                    chunk['quality_analysis'] = analysis
                    chunk['quality_score'] = analysis['quality_score']
                    chunk['quality_grade'] = analysis['quality_grade']
                    
                    # Collect batch statistics
                    batch_analysis['quality_scores'].append(analysis['quality_score'])
                    batch_analysis['issues_count'] += len(analysis['issues'])
                    batch_analysis['warnings_count'] += len(analysis['warnings'])
                    
                    # Collect unique recommendations
                    for rec in analysis['recommendations']:
                        if rec not in batch_analysis['recommendations']:
                            batch_analysis['recommendations'].append(rec)
                
                batch_analysis['analyzed_chunks'] = i + 1
            
            # Calculate batch statistics
            if batch_analysis['quality_scores']:
                batch_analysis['avg_quality'] = statistics.mean(batch_analysis['quality_scores'])
                
                # Quality distribution
                grades = [self.get_quality_grade(score) for score in batch_analysis['quality_scores']]
                batch_analysis['quality_distribution'] = {
                    grade: grades.count(grade) for grade in set(grades)
                }
            
            batch_analysis['analysis_complete'] = True
            return batch_analysis
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {str(e)}")
            batch_analysis['error'] = str(e)
            return batch_analysis
    
    def suggest_optimizations(self, analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate smart optimization suggestions"""
        suggestions = []
        
        if analysis['quality_score'] < 0.6:
            suggestions.append({
                'type': 'quality',
                'priority': 'high',
                'message': 'Overall quality is low',
                'action': 'Review and improve content structure',
                'button_text': 'Improve Quality'
            })
        
        if analysis['dialogue_potential'] < 0.3:
            suggestions.append({
                'type': 'enhancement',
                'priority': 'medium',
                'message': 'Low dialogue potential detected',
                'action': 'Use narrative-to-dialogue enhancement mode',
                'button_text': 'Enhance for Dialogue'
            })
        
        if analysis['word_count'] > 500:
            suggestions.append({
                'type': 'chunking',
                'priority': 'medium',
                'message': 'Large chunk detected',
                'action': 'Split into smaller chunks for better processing',
                'button_text': 'Optimize Chunk Size'
            })
        
        if analysis['coherence_score'] < 0.5:
            suggestions.append({
                'type': 'structure',
                'priority': 'high',
                'message': 'Low coherence detected',
                'action': 'Review content structure and add transitions',
                'button_text': 'Improve Structure'
            })
        
        return suggestions
    
    def validate_export_quality(self, data: List[Dict[str, Any]], export_format: str) -> Dict[str, Any]:
        """Validate quality before export"""
        validation_results = {
            'is_valid': True,
            'total_records': len(data),
            'quality_passed': 0,
            'quality_failed': 0,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'quality_distribution': {},
            'avg_quality': 0
        }
        
        try:
            quality_scores = []
            
            for i, record in enumerate(data):
                quality_score = record.get('quality_score', 0)
                quality_scores.append(quality_score)
                
                if quality_score >= 0.6:
                    validation_results['quality_passed'] += 1
                else:
                    validation_results['quality_failed'] += 1
                    validation_results['warnings'].append(f"Record {i+1}: Low quality score ({quality_score:.2f})")
            
            # Calculate statistics
            if quality_scores:
                validation_results['avg_quality'] = statistics.mean(quality_scores)
                
                # Quality distribution
                grades = [self.get_quality_grade(score) for score in quality_scores]
                validation_results['quality_distribution'] = {
                    grade: grades.count(grade) for grade in set(grades)
                }
            
            # Quality recommendations
            if validation_results['quality_failed'] > 0:
                failure_rate = validation_results['quality_failed'] / validation_results['total_records']
                if failure_rate > 0.3:
                    validation_results['recommendations'].append("Consider improving content quality before export")
                    validation_results['recommendations'].append("Review and enhance low-quality chunks")
            
            # Format-specific validation
            if export_format == 'jsonl':
                for i, record in enumerate(data):
                    if not record.get('content'):
                        validation_results['errors'].append(f"Record {i+1}: Missing content field")
                        validation_results['is_valid'] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Export quality validation failed: {str(e)}")
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            validation_results['is_valid'] = False
            return validation_results
    
    def get_quality_metrics_summary(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get summary of quality metrics for dashboard"""
        if not chunks:
            return {'total_chunks': 0}
        
        try:
            quality_scores = [chunk.get('quality_score', 0) for chunk in chunks if chunk.get('quality_score') is not None]
            
            if not quality_scores:
                return {'total_chunks': len(chunks), 'analyzed': 0}
            
            summary = {
                'total_chunks': len(chunks),
                'analyzed': len(quality_scores),
                'avg_quality': statistics.mean(quality_scores),
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores),
                'quality_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                'excellent_count': sum(1 for score in quality_scores if score >= 0.9),
                'good_count': sum(1 for score in quality_scores if 0.7 <= score < 0.9),
                'fair_count': sum(1 for score in quality_scores if 0.5 <= score < 0.7),
                'poor_count': sum(1 for score in quality_scores if score < 0.5),
            }
            
            # Calculate percentages
            total = len(quality_scores)
            summary['excellent_pct'] = (summary['excellent_count'] / total) * 100
            summary['good_pct'] = (summary['good_count'] / total) * 100
            summary['fair_pct'] = (summary['fair_count'] / total) * 100
            summary['poor_pct'] = (summary['poor_count'] / total) * 100
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Quality metrics summary failed: {str(e)}")
            return {'total_chunks': len(chunks), 'error': str(e)}

# Global instance
quality_monitor = QualityMonitor()

