"""
Enhanced spaCy Processor Module
Provides advanced NLP analysis using spaCy for intelligent content preparation
Replaces transformer-heavy dependencies with efficient CPU-only processing
"""

import spacy
from typing import Dict, List, Tuple, Any, Optional
import re
from collections import defaultdict, Counter
import logging

class EnhancedSpacyProcessor:
    """
    Advanced spaCy-based content processor for intelligent text analysis
    Provides entity extraction, linguistic analysis, and context understanding
    """
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the enhanced spaCy processor"""
        self.model_name = model_name
        self.nlp = None
        self.logger = logging.getLogger(__name__)
        self._load_model()
    
    def _load_model(self):
        """Load spaCy model with error handling"""
        try:
            self.nlp = spacy.load(self.model_name)
            self.logger.info(f"Successfully loaded spaCy model: {self.model_name}")
        except OSError:
            self.logger.warning(f"Model {self.model_name} not found, downloading...")
            try:
                spacy.cli.download(self.model_name)
                self.nlp = spacy.load(self.model_name)
                self.logger.info(f"Downloaded and loaded spaCy model: {self.model_name}")
            except Exception as e:
                self.logger.error(f"Failed to load spaCy model: {e}")
                raise
    
    def analyze_content(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive content analysis using spaCy
        Returns detailed linguistic and semantic information
        """
        if not text or not text.strip():
            return self._empty_analysis()
        
        try:
            doc = self.nlp(text)
            
            analysis = {
                'entities': self._extract_entities(doc),
                'linguistic_features': self._analyze_linguistic_features(doc),
                'structure': self._analyze_structure(doc),
                'topics': self._extract_topics(doc),
                'sentiment': self._analyze_sentiment(doc),
                'complexity': self._assess_complexity(doc),
                'key_phrases': self._extract_key_phrases(doc),
                'context_markers': self._identify_context_markers(doc)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing content: {e}")
            return self._empty_analysis()
    
    def _extract_entities(self, doc) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and categorize named entities"""
        entities = defaultdict(list)
        
        for ent in doc.ents:
            entity_info = {
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_),
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'confidence', 0.9)  # Default confidence
            }
            entities[ent.label_].append(entity_info)
        
        return dict(entities)
    
    def _analyze_linguistic_features(self, doc) -> Dict[str, Any]:
        """Analyze linguistic features of the text"""
        features = {
            'sentence_count': len(list(doc.sents)),
            'token_count': len(doc),
            'word_count': len([token for token in doc if not token.is_punct and not token.is_space]),
            'avg_sentence_length': 0,
            'pos_distribution': {},
            'dependency_patterns': [],
            'readability_indicators': {}
        }
        
        # Calculate average sentence length
        sentences = list(doc.sents)
        if sentences:
            features['avg_sentence_length'] = sum(len(sent) for sent in sentences) / len(sentences)
        
        # POS tag distribution
        pos_counts = Counter(token.pos_ for token in doc if not token.is_punct and not token.is_space)
        total_words = sum(pos_counts.values())
        if total_words > 0:
            features['pos_distribution'] = {pos: count/total_words for pos, count in pos_counts.items()}
        
        # Readability indicators
        features['readability_indicators'] = {
            'complex_words': len([token for token in doc if len(token.text) > 6 and token.is_alpha]),
            'unique_words': len(set(token.lemma_.lower() for token in doc if token.is_alpha)),
            'punctuation_density': len([token for token in doc if token.is_punct]) / len(doc) if len(doc) > 0 else 0
        }
        
        return features
    
    def _analyze_structure(self, doc) -> Dict[str, Any]:
        """Analyze document structure and organization"""
        sentences = list(doc.sents)
        
        structure = {
            'paragraph_breaks': self._identify_paragraph_breaks(doc.text),
            'sentence_types': self._classify_sentence_types(sentences),
            'discourse_markers': self._find_discourse_markers(doc),
            'question_answer_pairs': self._identify_qa_patterns(sentences),
            'dialogue_indicators': self._find_dialogue_indicators(doc),
            'list_structures': self._identify_lists(doc.text)
        }
        
        return structure
    
    def _extract_topics(self, doc) -> List[Dict[str, Any]]:
        """Extract potential topics using noun phrases and entities"""
        topics = []
        
        # Extract noun phrases as potential topics
        noun_phrases = [chunk.text.lower().strip() for chunk in doc.noun_chunks 
                       if len(chunk.text.strip()) > 2 and not chunk.root.is_stop]
        
        # Count frequency and filter
        phrase_counts = Counter(noun_phrases)
        
        for phrase, count in phrase_counts.most_common(10):
            topics.append({
                'phrase': phrase,
                'frequency': count,
                'type': 'noun_phrase',
                'importance': min(count / len(noun_phrases), 1.0) if noun_phrases else 0
            })
        
        return topics
    
    def _analyze_sentiment(self, doc) -> Dict[str, Any]:
        """Basic sentiment analysis using linguistic cues"""
        # Simple rule-based sentiment (can be enhanced with sentiment models)
        positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'love', 'like', 'enjoy']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'horrible', 'worst']
        
        tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
        
        positive_count = sum(1 for token in tokens if token in positive_words)
        negative_count = sum(1 for token in tokens if token in negative_words)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            polarity = 0.0
        else:
            polarity = (positive_count - negative_count) / total_sentiment_words
        
        return {
            'polarity': polarity,
            'positive_indicators': positive_count,
            'negative_indicators': negative_count,
            'subjectivity': total_sentiment_words / len(tokens) if tokens else 0
        }
    
    def _assess_complexity(self, doc) -> Dict[str, Any]:
        """Assess text complexity using various metrics"""
        sentences = list(doc.sents)
        words = [token for token in doc if token.is_alpha]
        
        complexity = {
            'lexical_diversity': 0,
            'syntactic_complexity': 0,
            'semantic_density': 0,
            'overall_score': 0
        }
        
        if words:
            # Lexical diversity (type-token ratio)
            unique_words = set(token.lemma_.lower() for token in words)
            complexity['lexical_diversity'] = len(unique_words) / len(words)
            
            # Syntactic complexity (average dependency depth)
            if sentences:
                avg_depth = sum(self._calculate_dependency_depth(sent.root) for sent in sentences) / len(sentences)
                complexity['syntactic_complexity'] = min(avg_depth / 5.0, 1.0)  # Normalize
            
            # Semantic density (entities and noun phrases per word)
            entities_count = len([ent for ent in doc.ents])
            noun_phrases_count = len(list(doc.noun_chunks))
            complexity['semantic_density'] = (entities_count + noun_phrases_count) / len(words)
            
            # Overall complexity score
            complexity['overall_score'] = (
                complexity['lexical_diversity'] * 0.3 +
                complexity['syntactic_complexity'] * 0.4 +
                complexity['semantic_density'] * 0.3
            )
        
        return complexity
    
    def _extract_key_phrases(self, doc) -> List[Dict[str, Any]]:
        """Extract key phrases using linguistic patterns"""
        key_phrases = []
        
        # Extract noun phrases with high importance
        for chunk in doc.noun_chunks:
            if len(chunk.text.strip()) > 3 and not chunk.root.is_stop:
                importance = self._calculate_phrase_importance(chunk)
                if importance > 0.3:  # Threshold for key phrases
                    key_phrases.append({
                        'text': chunk.text.strip(),
                        'importance': importance,
                        'type': 'noun_phrase',
                        'pos_pattern': ' '.join([token.pos_ for token in chunk])
                    })
        
        # Sort by importance
        key_phrases.sort(key=lambda x: x['importance'], reverse=True)
        return key_phrases[:15]  # Top 15 key phrases
    
    def _identify_context_markers(self, doc) -> Dict[str, List[str]]:
        """Identify context markers that help with content understanding"""
        markers = {
            'temporal': [],
            'causal': [],
            'comparative': [],
            'emphasis': [],
            'transition': []
        }
        
        # Define marker patterns
        temporal_markers = ['when', 'after', 'before', 'during', 'while', 'since', 'until']
        causal_markers = ['because', 'since', 'therefore', 'thus', 'consequently', 'as a result']
        comparative_markers = ['however', 'but', 'although', 'whereas', 'on the other hand']
        emphasis_markers = ['indeed', 'certainly', 'definitely', 'absolutely', 'particularly']
        transition_markers = ['furthermore', 'moreover', 'additionally', 'meanwhile', 'finally']
        
        text_lower = doc.text.lower()
        
        for marker in temporal_markers:
            if marker in text_lower:
                markers['temporal'].append(marker)
        
        for marker in causal_markers:
            if marker in text_lower:
                markers['causal'].append(marker)
        
        for marker in comparative_markers:
            if marker in text_lower:
                markers['comparative'].append(marker)
        
        for marker in emphasis_markers:
            if marker in text_lower:
                markers['emphasis'].append(marker)
        
        for marker in transition_markers:
            if marker in text_lower:
                markers['transition'].append(marker)
        
        return markers
    
    def _calculate_dependency_depth(self, token, depth=0):
        """Calculate the maximum dependency depth from a token"""
        if not list(token.children):
            return depth
        return max(self._calculate_dependency_depth(child, depth + 1) for child in token.children)
    
    def _calculate_phrase_importance(self, chunk):
        """Calculate importance score for a phrase"""
        importance = 0.5  # Base importance
        
        # Boost for entities
        if any(token.ent_type_ for token in chunk):
            importance += 0.3
        
        # Boost for proper nouns
        if any(token.pos_ == 'PROPN' for token in chunk):
            importance += 0.2
        
        # Reduce for very common words
        if chunk.root.is_stop:
            importance -= 0.2
        
        # Boost for longer phrases (more specific)
        if len(chunk) > 2:
            importance += 0.1
        
        return max(0, min(1, importance))
    
    def _identify_paragraph_breaks(self, text: str) -> List[int]:
        """Identify paragraph break positions"""
        breaks = []
        lines = text.split('\n')
        position = 0
        
        for i, line in enumerate(lines):
            if i > 0 and (not line.strip() or line.strip().startswith(('•', '-', '1.', '2.'))):
                breaks.append(position)
            position += len(line) + 1  # +1 for newline
        
        return breaks
    
    def _classify_sentence_types(self, sentences) -> Dict[str, int]:
        """Classify sentences by type"""
        types = {'declarative': 0, 'interrogative': 0, 'imperative': 0, 'exclamatory': 0}
        
        for sent in sentences:
            text = sent.text.strip()
            if text.endswith('?'):
                types['interrogative'] += 1
            elif text.endswith('!'):
                types['exclamatory'] += 1
            elif sent[0].pos_ == 'VERB' and sent[0].tag_ in ['VB', 'VBP']:
                types['imperative'] += 1
            else:
                types['declarative'] += 1
        
        return types
    
    def _find_discourse_markers(self, doc) -> List[str]:
        """Find discourse markers in the text"""
        markers = []
        discourse_words = ['however', 'therefore', 'furthermore', 'moreover', 'consequently', 
                          'nevertheless', 'meanwhile', 'subsequently', 'additionally']
        
        for token in doc:
            if token.lemma_.lower() in discourse_words:
                markers.append(token.text)
        
        return list(set(markers))
    
    def _identify_qa_patterns(self, sentences) -> List[Dict[str, Any]]:
        """Identify potential question-answer patterns"""
        qa_pairs = []
        
        for i, sent in enumerate(sentences):
            if sent.text.strip().endswith('?'):
                # Look for answer in next sentence
                if i + 1 < len(sentences):
                    qa_pairs.append({
                        'question': sent.text.strip(),
                        'potential_answer': sentences[i + 1].text.strip(),
                        'confidence': 0.7
                    })
        
        return qa_pairs
    
    def _find_dialogue_indicators(self, doc) -> Dict[str, Any]:
        """Find indicators of dialogue or conversation"""
        indicators = {
            'quotation_marks': doc.text.count('"') + doc.text.count("'"),
            'speaker_tags': [],
            'dialogue_verbs': []
        }
        
        dialogue_verbs = ['said', 'asked', 'replied', 'answered', 'explained', 'stated']
        
        for token in doc:
            if token.lemma_.lower() in dialogue_verbs:
                indicators['dialogue_verbs'].append(token.text)
        
        # Simple speaker tag detection
        lines = doc.text.split('\n')
        for line in lines:
            if ':' in line and len(line.split(':')[0].split()) <= 3:
                potential_speaker = line.split(':')[0].strip()
                if potential_speaker and potential_speaker[0].isupper():
                    indicators['speaker_tags'].append(potential_speaker)
        
        return indicators
    
    def _identify_lists(self, text: str) -> Dict[str, Any]:
        """Identify list structures in the text"""
        lists = {
            'bullet_points': [],
            'numbered_lists': [],
            'definition_lists': []
        }
        
        lines = text.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(('•', '-', '*')):
                lists['bullet_points'].append(stripped)
            elif re.match(r'^\d+\.', stripped):
                lists['numbered_lists'].append(stripped)
            elif ':' in stripped and not stripped.endswith('?'):
                lists['definition_lists'].append(stripped)
        
        return lists
    
    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure"""
        return {
            'entities': {},
            'linguistic_features': {},
            'structure': {},
            'topics': [],
            'sentiment': {'polarity': 0, 'subjectivity': 0},
            'complexity': {'overall_score': 0},
            'key_phrases': [],
            'context_markers': {}
        }
    
    def get_smart_chunks(self, text: str, max_chunk_size: int = 1000) -> List[Dict[str, Any]]:
        """
        Create intelligent chunks using spaCy's linguistic understanding
        Preserves sentence boundaries and maintains context
        """
        if not text or not text.strip():
            return []
        
        try:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            chunks = []
            current_chunk = ""
            current_sentences = []
            
            for sent in sentences:
                sent_text = sent.text.strip()
                
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sent_text) > max_chunk_size and current_chunk:
                    # Create chunk with current content
                    chunk_analysis = self.analyze_content(current_chunk)
                    chunks.append({
                        'text': current_chunk.strip(),
                        'sentence_count': len(current_sentences),
                        'analysis': chunk_analysis,
                        'chunk_id': len(chunks) + 1
                    })
                    
                    # Start new chunk
                    current_chunk = sent_text
                    current_sentences = [sent]
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sent_text
                    else:
                        current_chunk = sent_text
                    current_sentences.append(sent)
            
            # Add final chunk if it has content
            if current_chunk.strip():
                chunk_analysis = self.analyze_content(current_chunk)
                chunks.append({
                    'text': current_chunk.strip(),
                    'sentence_count': len(current_sentences),
                    'analysis': chunk_analysis,
                    'chunk_id': len(chunks) + 1
                })
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error creating smart chunks: {e}")
            # Fallback to simple chunking
            return self._simple_chunk_fallback(text, max_chunk_size)
    
    def _simple_chunk_fallback(self, text: str, max_chunk_size: int) -> List[Dict[str, Any]]:
        """Simple fallback chunking if spaCy processing fails"""
        chunks = []
        words = text.split()
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > max_chunk_size:
                if len(current_chunk) > 1:
                    current_chunk.pop()  # Remove last word
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        'text': chunk_text,
                        'sentence_count': chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?'),
                        'analysis': self._empty_analysis(),
                        'chunk_id': len(chunks) + 1
                    })
                    current_chunk = [word]
                else:
                    # Single word is too long, include it anyway
                    chunks.append({
                        'text': word,
                        'sentence_count': 1,
                        'analysis': self._empty_analysis(),
                        'chunk_id': len(chunks) + 1
                    })
                    current_chunk = []
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'sentence_count': chunk_text.count('.') + chunk_text.count('!') + chunk_text.count('?'),
                'analysis': self._empty_analysis(),
                'chunk_id': len(chunks) + 1
            })
        
        return chunks

