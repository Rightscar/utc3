"""
Advanced Semantic Understanding Engine
=====================================

This module implements sophisticated NLP capabilities that go beyond keyword matching
to provide true semantic understanding and context-aware processing.

Features:
- Semantic similarity analysis using sentence transformers
- Named entity recognition and relationship mapping
- Concept extraction and theme identification
- Narrative flow analysis
- Contextual relevance scoring
- Multi-dimensional semantic analysis
"""

import spacy
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
import re
import logging
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SemanticChunk:
    """Represents a semantically analyzed text chunk"""
    text: str
    start_pos: int
    end_pos: int
    semantic_score: float
    entities: List[Dict[str, Any]]
    concepts: List[str]
    themes: List[str]
    narrative_position: float
    context_relevance: float
    embedding: np.ndarray
    relationships: List[Dict[str, Any]]

@dataclass
class SemanticAnalysis:
    """Complete semantic analysis results"""
    chunks: List[SemanticChunk]
    global_themes: List[str]
    entity_graph: Dict[str, List[str]]
    narrative_arc: List[float]
    concept_clusters: Dict[str, List[str]]
    semantic_coherence: float

class SemanticUnderstandingEngine:
    """Advanced semantic understanding and analysis engine"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the semantic understanding engine"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded successfully")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using basic tokenization")
            self.nlp = None
        
        try:
            # Load sentence transformer model
            self.sentence_model = SentenceTransformer(model_name)
            logger.info(f"✅ Sentence transformer model '{model_name}' loaded")
        except Exception as e:
            logger.warning(f"⚠️ Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        # Initialize semantic analysis components
        self.entity_types = {
            'PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 
            'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 
            'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL'
        }
        
        # Concept extraction patterns
        self.concept_patterns = [
            r'\b(?:concept|idea|principle|theory|notion|belief)\s+of\s+(\w+)',
            r'\b(?:understanding|knowledge|awareness)\s+(?:of|about)\s+(\w+)',
            r'\b(?:philosophy|approach|method|strategy)\s+(?:of|for)\s+(\w+)',
            r'\b(?:nature|essence|core|heart)\s+of\s+(\w+)',
        ]
        
        # Theme identification keywords
        self.theme_keywords = {
            'consciousness': ['awareness', 'consciousness', 'mindfulness', 'perception', 'cognition'],
            'identity': ['self', 'identity', 'ego', 'personality', 'character'],
            'relationships': ['relationship', 'connection', 'bond', 'interaction', 'communication'],
            'growth': ['development', 'growth', 'evolution', 'progress', 'transformation'],
            'conflict': ['conflict', 'struggle', 'tension', 'opposition', 'challenge'],
            'resolution': ['resolution', 'solution', 'answer', 'conclusion', 'outcome'],
            'emotion': ['emotion', 'feeling', 'sentiment', 'mood', 'passion'],
            'knowledge': ['knowledge', 'learning', 'understanding', 'wisdom', 'insight'],
            'power': ['power', 'control', 'authority', 'influence', 'dominance'],
            'freedom': ['freedom', 'liberty', 'independence', 'autonomy', 'choice']
        }
    
    def analyze_semantic_structure(self, text: str, chunk_size: int = 500) -> SemanticAnalysis:
        """
        Perform comprehensive semantic analysis of text
        
        Args:
            text: Input text to analyze
            chunk_size: Target size for semantic chunks
            
        Returns:
            SemanticAnalysis object with complete analysis results
        """
        logger.info("🧠 Starting comprehensive semantic analysis...")
        
        # Step 1: Create semantic chunks
        chunks = self._create_semantic_chunks(text, chunk_size)
        
        # Step 2: Analyze each chunk
        analyzed_chunks = []
        for chunk in chunks:
            analyzed_chunk = self._analyze_chunk_semantics(chunk)
            analyzed_chunks.append(analyzed_chunk)
        
        # Step 3: Global analysis
        global_themes = self._extract_global_themes(analyzed_chunks)
        entity_graph = self._build_entity_graph(analyzed_chunks)
        narrative_arc = self._analyze_narrative_arc(analyzed_chunks)
        concept_clusters = self._cluster_concepts(analyzed_chunks)
        semantic_coherence = self._calculate_semantic_coherence(analyzed_chunks)
        
        analysis = SemanticAnalysis(
            chunks=analyzed_chunks,
            global_themes=global_themes,
            entity_graph=entity_graph,
            narrative_arc=narrative_arc,
            concept_clusters=concept_clusters,
            semantic_coherence=semantic_coherence
        )
        
        logger.info(f"✅ Semantic analysis complete: {len(analyzed_chunks)} chunks, {len(global_themes)} themes")
        return analysis
    
    def _create_semantic_chunks(self, text: str, target_size: int) -> List[Dict[str, Any]]:
        """Create semantically coherent chunks"""
        if not self.nlp:
            # Fallback to simple sentence-based chunking
            sentences = re.split(r'[.!?]+', text)
            chunks = []
            current_chunk = ""
            start_pos = 0
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > target_size and current_chunk:
                    chunks.append({
                        'text': current_chunk.strip(),
                        'start_pos': start_pos,
                        'end_pos': start_pos + len(current_chunk)
                    })
                    start_pos += len(current_chunk)
                    current_chunk = sentence
                else:
                    current_chunk += sentence + ". "
            
            if current_chunk.strip():
                chunks.append({
                    'text': current_chunk.strip(),
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(current_chunk)
                })
            
            return chunks
        
        # Advanced semantic chunking with spaCy
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_size = 0
        start_pos = 0
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if current_size + len(sent_text) > target_size and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'start_pos': start_pos,
                    'end_pos': start_pos + len(chunk_text),
                    'sentences': current_chunk.copy()
                })
                
                # Reset for next chunk
                start_pos += len(chunk_text)
                current_chunk = [sent_text]
                current_size = len(sent_text)
            else:
                current_chunk.append(sent_text)
                current_size += len(sent_text)
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'start_pos': start_pos,
                'end_pos': start_pos + len(chunk_text),
                'sentences': current_chunk.copy()
            })
        
        return chunks
    
    def _analyze_chunk_semantics(self, chunk: Dict[str, Any]) -> SemanticChunk:
        """Perform detailed semantic analysis on a single chunk"""
        text = chunk['text']
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Extract concepts
        concepts = self._extract_concepts(text)
        
        # Identify themes
        themes = self._identify_themes(text)
        
        # Calculate semantic score
        semantic_score = self._calculate_semantic_score(text, entities, concepts, themes)
        
        # Generate embedding
        embedding = self._generate_embedding(text)
        
        # Analyze relationships
        relationships = self._analyze_relationships(text, entities)
        
        # Calculate narrative position (0.0 = beginning, 1.0 = end)
        narrative_position = chunk['start_pos'] / (chunk['end_pos'] + 1)
        
        # Calculate context relevance
        context_relevance = self._calculate_context_relevance(text, concepts, themes)
        
        return SemanticChunk(
            text=text,
            start_pos=chunk['start_pos'],
            end_pos=chunk['end_pos'],
            semantic_score=semantic_score,
            entities=entities,
            concepts=concepts,
            themes=themes,
            narrative_position=narrative_position,
            context_relevance=context_relevance,
            embedding=embedding,
            relationships=relationships
        )
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities with enhanced analysis"""
        entities = []
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char,
                        'confidence': getattr(ent, 'confidence', 0.8)
                    })
        else:
            # Fallback entity extraction using patterns
            patterns = {
                'PERSON': r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',
                'ORG': r'\b[A-Z][a-z]+\s+(?:Inc|Corp|Ltd|Company|Organization)\b',
                'GPE': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:City|State|Country)\b'
            }
            
            for label, pattern in patterns.items():
                for match in re.finditer(pattern, text):
                    entities.append({
                        'text': match.group(),
                        'label': label,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.6
                    })
        
        return entities
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        concepts = []
        
        # Pattern-based concept extraction
        for pattern in self.concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        # Noun phrase extraction
        if self.nlp:
            doc = self.nlp(text)
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Multi-word concepts
                    concepts.append(chunk.text.lower())
        
        # Remove duplicates and filter
        concepts = list(set([c.lower().strip() for c in concepts if len(c) > 2]))
        
        return concepts[:10]  # Top 10 concepts
    
    def _identify_themes(self, text: str) -> List[str]:
        """Identify thematic elements in text"""
        text_lower = text.lower()
        themes = []
        
        for theme, keywords in self.theme_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                themes.append({
                    'theme': theme,
                    'score': score,
                    'keywords_found': [kw for kw in keywords if kw in text_lower]
                })
        
        # Sort by score and return top themes
        themes.sort(key=lambda x: x['score'], reverse=True)
        return [t['theme'] for t in themes[:5]]
    
    def _calculate_semantic_score(self, text: str, entities: List[Dict], 
                                concepts: List[str], themes: List[str]) -> float:
        """Calculate overall semantic richness score"""
        # Base score from text length and complexity
        words = len(text.split())
        sentences = len(re.split(r'[.!?]+', text))
        avg_sentence_length = words / max(sentences, 1)
        
        # Complexity factors
        entity_density = len(entities) / max(words, 1)
        concept_density = len(concepts) / max(words, 1)
        theme_diversity = len(themes) / 10.0  # Normalize to max 10 themes
        
        # Vocabulary diversity
        unique_words = len(set(text.lower().split()))
        vocab_diversity = unique_words / max(words, 1)
        
        # Combine factors
        semantic_score = (
            min(avg_sentence_length / 20.0, 1.0) * 0.2 +  # Sentence complexity
            min(entity_density * 100, 1.0) * 0.3 +        # Entity richness
            min(concept_density * 50, 1.0) * 0.2 +         # Concept density
            theme_diversity * 0.2 +                        # Theme diversity
            vocab_diversity * 0.1                          # Vocabulary diversity
        )
        
        return min(semantic_score, 1.0)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text"""
        if self.sentence_model:
            try:
                embedding = self.sentence_model.encode(text)
                return embedding
            except Exception as e:
                logger.warning(f"Could not generate embedding: {e}")
        
        # Fallback: simple word frequency vector
        words = text.lower().split()
        word_counts = Counter(words)
        # Create a simple 100-dimensional vector
        vector = np.zeros(100)
        for i, (word, count) in enumerate(word_counts.most_common(100)):
            if i < 100:
                vector[i] = count
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def _analyze_relationships(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """Analyze relationships between entities"""
        relationships = []
        
        if not self.nlp or len(entities) < 2:
            return relationships
        
        doc = self.nlp(text)
        
        # Simple relationship extraction based on proximity and dependency parsing
        for i, ent1 in enumerate(entities):
            for j, ent2 in enumerate(entities[i+1:], i+1):
                # Check if entities appear in same sentence
                for sent in doc.sents:
                    if (ent1['start'] >= sent.start_char and ent1['end'] <= sent.end_char and
                        ent2['start'] >= sent.start_char and ent2['end'] <= sent.end_char):
                        
                        relationships.append({
                            'entity1': ent1['text'],
                            'entity2': ent2['text'],
                            'type': 'co_occurrence',
                            'context': sent.text,
                            'confidence': 0.7
                        })
        
        return relationships
    
    def _calculate_context_relevance(self, text: str, concepts: List[str], themes: List[str]) -> float:
        """Calculate how relevant this chunk is to the overall context"""
        # Simple relevance based on concept and theme density
        words = len(text.split())
        concept_mentions = sum(1 for concept in concepts if concept.lower() in text.lower())
        theme_mentions = sum(1 for theme in themes 
                           if any(keyword in text.lower() 
                                for keyword in self.theme_keywords.get(theme, [])))
        
        relevance = (concept_mentions + theme_mentions) / max(words / 100, 1)
        return min(relevance, 1.0)
    
    def _extract_global_themes(self, chunks: List[SemanticChunk]) -> List[str]:
        """Extract themes that appear across multiple chunks"""
        theme_counts = Counter()
        
        for chunk in chunks:
            for theme in chunk.themes:
                theme_counts[theme] += 1
        
        # Return themes that appear in multiple chunks
        global_themes = [theme for theme, count in theme_counts.items() 
                        if count >= max(2, len(chunks) * 0.2)]
        
        return global_themes
    
    def _build_entity_graph(self, chunks: List[SemanticChunk]) -> Dict[str, List[str]]:
        """Build a graph of entity relationships"""
        entity_graph = defaultdict(list)
        
        for chunk in chunks:
            for relationship in chunk.relationships:
                entity1 = relationship['entity1']
                entity2 = relationship['entity2']
                entity_graph[entity1].append(entity2)
                entity_graph[entity2].append(entity1)
        
        # Remove duplicates
        for entity in entity_graph:
            entity_graph[entity] = list(set(entity_graph[entity]))
        
        return dict(entity_graph)
    
    def _analyze_narrative_arc(self, chunks: List[SemanticChunk]) -> List[float]:
        """Analyze the narrative progression through the text"""
        if not chunks:
            return []
        
        # Calculate semantic density progression
        arc = []
        for chunk in chunks:
            # Combine various factors for narrative intensity
            intensity = (
                chunk.semantic_score * 0.4 +
                len(chunk.entities) / 10.0 * 0.3 +
                len(chunk.concepts) / 10.0 * 0.2 +
                chunk.context_relevance * 0.1
            )
            arc.append(min(intensity, 1.0))
        
        return arc
    
    def _cluster_concepts(self, chunks: List[SemanticChunk]) -> Dict[str, List[str]]:
        """Cluster related concepts together"""
        all_concepts = []
        for chunk in chunks:
            all_concepts.extend(chunk.concepts)
        
        if not all_concepts:
            return {}
        
        # Simple clustering based on co-occurrence
        concept_counts = Counter(all_concepts)
        frequent_concepts = [concept for concept, count in concept_counts.items() if count >= 2]
        
        if len(frequent_concepts) < 2:
            return {'main_concepts': frequent_concepts}
        
        # Group concepts that appear together frequently
        clusters = defaultdict(list)
        
        for chunk in chunks:
            chunk_concepts = [c for c in chunk.concepts if c in frequent_concepts]
            if len(chunk_concepts) >= 2:
                # All concepts in this chunk are related
                cluster_key = f"cluster_{len(clusters)}"
                clusters[cluster_key].extend(chunk_concepts)
        
        # Remove duplicates and small clusters
        final_clusters = {}
        for cluster_name, concepts in clusters.items():
            unique_concepts = list(set(concepts))
            if len(unique_concepts) >= 2:
                final_clusters[cluster_name] = unique_concepts
        
        return final_clusters
    
    def _calculate_semantic_coherence(self, chunks: List[SemanticChunk]) -> float:
        """Calculate overall semantic coherence of the text"""
        if len(chunks) < 2:
            return 1.0
        
        if not self.sentence_model:
            # Fallback: simple coherence based on theme consistency
            all_themes = []
            for chunk in chunks:
                all_themes.extend(chunk.themes)
            
            if not all_themes:
                return 0.5
            
            theme_counts = Counter(all_themes)
            most_common_theme_count = theme_counts.most_common(1)[0][1]
            coherence = most_common_theme_count / len(chunks)
            return min(coherence, 1.0)
        
        # Calculate coherence using embeddings
        embeddings = [chunk.embedding for chunk in chunks if chunk.embedding is not None]
        
        if len(embeddings) < 2:
            return 0.5
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
            similarities.append(sim)
        
        # Average similarity between consecutive chunks
        coherence = np.mean(similarities)
        return max(0.0, min(coherence, 1.0))
    
    def find_semantic_boundaries(self, text: str, min_chunk_size: int = 200) -> List[int]:
        """Find optimal semantic boundaries for chunking"""
        if not self.nlp:
            # Fallback to sentence boundaries
            sentences = list(re.finditer(r'[.!?]+', text))
            boundaries = [0]
            current_size = 0
            
            for match in sentences:
                pos = match.end()
                if current_size >= min_chunk_size:
                    boundaries.append(pos)
                    current_size = 0
                else:
                    current_size = pos - boundaries[-1]
            
            if boundaries[-1] < len(text):
                boundaries.append(len(text))
            
            return boundaries
        
        doc = self.nlp(text)
        boundaries = [0]
        current_size = 0
        
        for sent in doc.sents:
            sent_end = sent.end_char
            chunk_size = sent_end - boundaries[-1]
            
            if chunk_size >= min_chunk_size:
                # Check if this is a good semantic boundary
                if self._is_semantic_boundary(sent):
                    boundaries.append(sent_end)
            
        if boundaries[-1] < len(text):
            boundaries.append(len(text))
        
        return boundaries
    
    def _is_semantic_boundary(self, sentence) -> bool:
        """Determine if a sentence represents a good semantic boundary"""
        text = sentence.text.lower()
        
        # Indicators of semantic boundaries
        boundary_indicators = [
            'however', 'meanwhile', 'furthermore', 'moreover', 'in addition',
            'on the other hand', 'in contrast', 'nevertheless', 'therefore',
            'consequently', 'as a result', 'in conclusion', 'finally',
            'first', 'second', 'third', 'next', 'then', 'later',
            'chapter', 'section', 'part'
        ]
        
        return any(indicator in text for indicator in boundary_indicators)
    
    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        if self.sentence_model:
            try:
                embeddings = self.sentence_model.encode([text1, text2])
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return float(similarity)
            except Exception as e:
                logger.warning(f"Could not calculate semantic similarity: {e}")
        
        # Fallback: simple word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def extract_key_phrases(self, text: str, max_phrases: int = 10) -> List[Dict[str, Any]]:
        """Extract key phrases with semantic importance scores"""
        key_phrases = []
        
        if self.nlp:
            doc = self.nlp(text)
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) >= 2:  # Multi-word phrases
                    importance = self._calculate_phrase_importance(chunk, doc)
                    key_phrases.append({
                        'phrase': chunk.text,
                        'importance': importance,
                        'pos': chunk.start_char,
                        'type': 'noun_phrase'
                    })
            
            # Extract verb phrases
            for token in doc:
                if token.pos_ == 'VERB' and token.head == token:
                    phrase_tokens = [token]
                    for child in token.children:
                        if child.dep_ in ['dobj', 'prep', 'advmod']:
                            phrase_tokens.append(child)
                    
                    if len(phrase_tokens) > 1:
                        phrase_text = ' '.join([t.text for t in phrase_tokens])
                        importance = len(phrase_tokens) / 10.0  # Simple importance
                        key_phrases.append({
                            'phrase': phrase_text,
                            'importance': importance,
                            'pos': token.idx,
                            'type': 'verb_phrase'
                        })
        
        # Sort by importance and return top phrases
        key_phrases.sort(key=lambda x: x['importance'], reverse=True)
        return key_phrases[:max_phrases]
    
    def _calculate_phrase_importance(self, chunk, doc) -> float:
        """Calculate the semantic importance of a phrase"""
        # Factors: length, position, entity status, frequency
        length_score = min(len(chunk.text.split()) / 5.0, 1.0)
        
        # Check if phrase contains entities
        entity_score = 0.0
        for ent in doc.ents:
            if chunk.start <= ent.start < chunk.end or chunk.start < ent.end <= chunk.end:
                entity_score = 0.5
                break
        
        # Position score (middle of text is often more important)
        relative_pos = chunk.start_char / len(doc.text)
        position_score = 1.0 - abs(0.5 - relative_pos)
        
        importance = (length_score * 0.4 + entity_score * 0.4 + position_score * 0.2)
        return importance


def test_semantic_understanding():
    """Test the semantic understanding engine"""
    print("🧪 Testing Semantic Understanding Engine...")
    
    # Test text
    test_text = """
    Consciousness is a fascinating aspect of human experience. It involves our awareness 
    of ourselves and our environment. Many philosophers and scientists have studied 
    consciousness for centuries. The nature of consciousness remains one of the greatest 
    mysteries in science and philosophy.
    
    Self-awareness is a key component of consciousness. It allows us to reflect on our 
    thoughts and actions. This metacognitive ability sets humans apart from many other 
    species. However, the relationship between consciousness and the brain is still 
    not fully understood.
    
    Recent advances in neuroscience have provided new insights into consciousness. 
    Brain imaging techniques allow researchers to observe neural activity associated 
    with conscious experience. These studies suggest that consciousness emerges from 
    complex patterns of neural activity across different brain regions.
    """
    
    # Initialize engine
    engine = SemanticUnderstandingEngine()
    
    # Perform analysis
    analysis = engine.analyze_semantic_structure(test_text, chunk_size=300)
    
    # Display results
    print(f"\n📊 Analysis Results:")
    print(f"   Chunks: {len(analysis.chunks)}")
    print(f"   Global themes: {analysis.global_themes}")
    print(f"   Semantic coherence: {analysis.semantic_coherence:.3f}")
    print(f"   Concept clusters: {len(analysis.concept_clusters)}")
    
    for i, chunk in enumerate(analysis.chunks):
        print(f"\n📝 Chunk {i+1}:")
        print(f"   Semantic score: {chunk.semantic_score:.3f}")
        print(f"   Themes: {chunk.themes}")
        print(f"   Concepts: {chunk.concepts[:3]}...")  # Show first 3
        print(f"   Entities: {len(chunk.entities)}")
        print(f"   Context relevance: {chunk.context_relevance:.3f}")
    
    print("\n✅ Semantic understanding engine test completed!")

if __name__ == "__main__":
    test_semantic_understanding()

