"""
Context-Aware Chunking Engine
============================

This module implements intelligent chunking that understands narrative flow,
maintains context coherence, and adapts to content structure dynamically.

Features:
- Narrative flow analysis and preservation
- Context-aware boundary detection
- Adaptive chunk sizing based on content density
- Semantic coherence optimization
- Multi-level chunking (document, section, paragraph, sentence)
- Token budget optimization for AI processing
- Content type detection and specialized handling
"""

import re
import spacy
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import our semantic understanding engine
from .semantic_understanding_engine import SemanticUnderstandingEngine, SemanticChunk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContextChunk:
    """Represents a context-aware chunk with narrative understanding"""
    text: str
    start_pos: int
    end_pos: int
    chunk_id: str
    level: str  # 'document', 'section', 'paragraph', 'sentence'
    
    # Context information
    preceding_context: str
    following_context: str
    narrative_position: float  # 0.0 = beginning, 1.0 = end
    
    # Semantic properties
    semantic_density: float
    coherence_score: float
    topic_continuity: float
    
    # Structural properties
    sentence_count: int
    word_count: int
    token_count: int
    
    # Content analysis
    content_type: str  # 'narrative', 'dialogue', 'description', 'exposition'
    themes: List[str]
    entities: List[str]
    key_concepts: List[str]
    
    # Quality metrics
    readability_score: float
    information_density: float
    dialogue_potential: float
    
    # Relationships
    related_chunks: List[str]  # IDs of semantically related chunks
    dependency_chunks: List[str]  # IDs of chunks this depends on for context

@dataclass
class ChunkingStrategy:
    """Configuration for chunking strategy"""
    target_size: int = 500  # Target chunk size in words
    min_size: int = 100     # Minimum chunk size
    max_size: int = 1000    # Maximum chunk size
    
    # Token budget for AI processing
    token_budget: int = 1000  # Available tokens for AI processing
    prompt_tokens: int = 200  # Tokens reserved for prompt
    
    # Context preservation
    context_window: int = 100  # Words of context to preserve
    overlap_size: int = 50     # Overlap between chunks
    
    # Quality thresholds
    min_coherence: float = 0.6
    min_semantic_density: float = 0.4
    
    # Content type preferences
    preserve_dialogue: bool = True
    preserve_paragraphs: bool = True
    respect_sections: bool = True
    
    # Adaptive sizing
    adaptive_sizing: bool = True
    density_factor: float = 1.5  # Adjust size based on content density

class ContextAwareChunker:
    """Advanced chunking engine with narrative flow understanding"""
    
    def __init__(self, strategy: Optional[ChunkingStrategy] = None):
        """Initialize the context-aware chunker"""
        self.strategy = strategy or ChunkingStrategy()
        
        # Initialize semantic understanding engine
        self.semantic_engine = SemanticUnderstandingEngine()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ spaCy model loaded for context-aware chunking")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using basic processing")
            self.nlp = None
        
        # Content type patterns
        self.content_patterns = {
            'dialogue': [
                r'"[^"]*"',  # Quoted speech
                r"'[^']*'",  # Single quoted speech
                r'\b(?:said|asked|replied|answered|whispered|shouted)\b',
                r'\b(?:he said|she said|they said)\b'
            ],
            'description': [
                r'\b(?:looked|appeared|seemed|was|were)\b.*\b(?:like|as)\b',
                r'\b(?:tall|short|beautiful|ugly|old|young|large|small)\b',
                r'\b(?:color|colour|shape|size|texture)\b'
            ],
            'exposition': [
                r'\b(?:because|therefore|however|moreover|furthermore)\b',
                r'\b(?:first|second|third|finally|in conclusion)\b',
                r'\b(?:according to|research shows|studies indicate)\b'
            ],
            'narrative': [
                r'\b(?:then|next|after|before|while|during)\b',
                r'\b(?:suddenly|immediately|gradually|slowly)\b',
                r'\b(?:happened|occurred|took place|began|ended)\b'
            ]
        }
        
        # Narrative flow indicators
        self.flow_indicators = {
            'temporal': ['then', 'next', 'after', 'before', 'while', 'during', 'meanwhile'],
            'causal': ['because', 'therefore', 'consequently', 'as a result', 'due to'],
            'contrast': ['however', 'but', 'nevertheless', 'on the other hand', 'despite'],
            'addition': ['furthermore', 'moreover', 'in addition', 'also', 'besides'],
            'conclusion': ['finally', 'in conclusion', 'to summarize', 'therefore', 'thus']
        }
    
    def chunk_with_context(self, text: str, metadata: Optional[Dict] = None) -> List[ContextChunk]:
        """
        Perform context-aware chunking with narrative flow preservation
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata about the text
            
        Returns:
            List of ContextChunk objects with full context analysis
        """
        logger.info("🧠 Starting context-aware chunking with narrative flow analysis...")
        
        # Step 1: Analyze document structure
        document_structure = self._analyze_document_structure(text)
        
        # Step 2: Detect content types and narrative flow
        content_analysis = self._analyze_content_flow(text)
        
        # Step 3: Find optimal chunk boundaries
        boundaries = self._find_optimal_boundaries(text, document_structure, content_analysis)
        
        # Step 4: Create context-aware chunks
        chunks = self._create_context_chunks(text, boundaries, content_analysis)
        
        # Step 5: Optimize chunk relationships and dependencies
        chunks = self._optimize_chunk_relationships(chunks)
        
        # Step 6: Validate and adjust chunks
        chunks = self._validate_and_adjust_chunks(chunks, text)
        
        logger.info(f"✅ Context-aware chunking complete: {len(chunks)} chunks created")
        return chunks
    
    def _analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """Analyze the overall structure of the document"""
        structure = {
            'sections': [],
            'paragraphs': [],
            'sentences': [],
            'total_length': len(text),
            'estimated_reading_time': len(text.split()) / 200  # ~200 WPM
        }
        
        # Detect sections (chapters, parts, etc.)
        section_patterns = [
            r'^(Chapter|Part|Section)\s+\d+',
            r'^[A-Z][A-Z\s]{10,}$',  # ALL CAPS headings
            r'^\d+\.\s+[A-Z]',       # Numbered sections
            r'^#{1,6}\s+',           # Markdown headers
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if any(re.match(pattern, line, re.MULTILINE) for pattern in section_patterns):
                structure['sections'].append({
                    'title': line,
                    'line_number': i,
                    'position': text.find(line)
                })
        
        # Detect paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        current_pos = 0
        for para in paragraphs:
            if para.strip():
                structure['paragraphs'].append({
                    'text': para.strip(),
                    'start': current_pos,
                    'end': current_pos + len(para),
                    'word_count': len(para.split())
                })
            current_pos += len(para) + 2  # Account for newlines
        
        # Analyze sentences
        if self.nlp:
            doc = self.nlp(text)
            for sent in doc.sents:
                structure['sentences'].append({
                    'text': sent.text,
                    'start': sent.start_char,
                    'end': sent.end_char,
                    'word_count': len(sent.text.split())
                })
        else:
            # Fallback sentence detection
            sentences = re.split(r'[.!?]+', text)
            current_pos = 0
            for sent in sentences:
                if sent.strip():
                    structure['sentences'].append({
                        'text': sent.strip(),
                        'start': current_pos,
                        'end': current_pos + len(sent),
                        'word_count': len(sent.split())
                    })
                current_pos += len(sent) + 1
        
        return structure
    
    def _analyze_content_flow(self, text: str) -> Dict[str, Any]:
        """Analyze content types and narrative flow patterns"""
        analysis = {
            'content_types': {},
            'flow_patterns': {},
            'narrative_arc': [],
            'topic_transitions': [],
            'dialogue_sections': [],
            'description_sections': []
        }
        
        # Analyze content types
        for content_type, patterns in self.content_patterns.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text, re.IGNORECASE))
            analysis['content_types'][content_type] = matches
        
        # Analyze flow patterns
        for flow_type, indicators in self.flow_indicators.items():
            matches = 0
            positions = []
            for indicator in indicators:
                for match in re.finditer(r'\b' + re.escape(indicator) + r'\b', text, re.IGNORECASE):
                    matches += 1
                    positions.append(match.start())
            analysis['flow_patterns'][flow_type] = {
                'count': matches,
                'positions': positions
            }
        
        # Detect dialogue sections
        dialogue_pattern = r'"[^"]*"[^"]*(?:said|asked|replied|answered)'
        for match in re.finditer(dialogue_pattern, text, re.IGNORECASE):
            analysis['dialogue_sections'].append({
                'start': match.start(),
                'end': match.end(),
                'text': match.group()
            })
        
        # Analyze narrative arc (semantic density over position)
        if self.nlp:
            doc = self.nlp(text)
            chunk_size = len(text) // 10  # Divide into 10 segments
            for i in range(10):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, len(text))
                segment = text[start:end]
                
                # Calculate semantic density
                entities = len([ent for ent in doc.ents if start <= ent.start_char < end])
                words = len(segment.split())
                density = entities / max(words, 1) if words > 0 else 0
                
                analysis['narrative_arc'].append({
                    'position': i / 10.0,
                    'semantic_density': density,
                    'word_count': words
                })
        
        return analysis
    
    def _find_optimal_boundaries(self, text: str, structure: Dict, content_analysis: Dict) -> List[int]:
        """Find optimal chunk boundaries based on context and narrative flow"""
        boundaries = [0]  # Start with beginning of text
        
        # Strategy 1: Respect paragraph boundaries when possible
        if self.strategy.preserve_paragraphs:
            paragraph_boundaries = [p['start'] for p in structure['paragraphs']]
            paragraph_boundaries.append(len(text))
        else:
            paragraph_boundaries = []
        
        # Strategy 2: Respect section boundaries
        if self.strategy.respect_sections:
            section_boundaries = [s['position'] for s in structure['sections']]
        else:
            section_boundaries = []
        
        # Strategy 3: Find semantic boundaries
        semantic_boundaries = self._find_semantic_boundaries(text, content_analysis)
        
        # Combine all boundary candidates
        all_boundaries = set(paragraph_boundaries + section_boundaries + semantic_boundaries)
        all_boundaries.add(0)
        all_boundaries.add(len(text))
        all_boundaries = sorted(list(all_boundaries))
        
        # Select optimal boundaries based on target chunk size
        current_pos = 0
        target_size = self.strategy.target_size
        
        for boundary in all_boundaries:
            if boundary <= current_pos:
                continue
                
            chunk_size = self._estimate_word_count(text[current_pos:boundary])
            
            # Check if this boundary creates a good chunk
            if (chunk_size >= self.strategy.min_size and 
                chunk_size <= self.strategy.max_size):
                boundaries.append(boundary)
                current_pos = boundary
            elif chunk_size > self.strategy.max_size:
                # Need to split this section further
                sub_boundaries = self._split_large_section(
                    text[current_pos:boundary], current_pos, target_size
                )
                boundaries.extend(sub_boundaries)
                current_pos = boundary
        
        # Ensure we end at the text end
        if boundaries[-1] < len(text):
            boundaries.append(len(text))
        
        return boundaries
    
    def _find_semantic_boundaries(self, text: str, content_analysis: Dict) -> List[int]:
        """Find boundaries based on semantic transitions"""
        boundaries = []
        
        # Use flow pattern positions as potential boundaries
        for flow_type, data in content_analysis['flow_patterns'].items():
            if flow_type in ['contrast', 'conclusion', 'temporal']:
                boundaries.extend(data['positions'])
        
        # Add dialogue boundaries
        for dialogue in content_analysis['dialogue_sections']:
            boundaries.extend([dialogue['start'], dialogue['end']])
        
        # Use semantic understanding engine for additional boundaries
        if hasattr(self.semantic_engine, 'find_semantic_boundaries'):
            semantic_bounds = self.semantic_engine.find_semantic_boundaries(
                text, self.strategy.min_size * 4  # Convert words to chars (rough estimate)
            )
            boundaries.extend(semantic_bounds)
        
        return sorted(list(set(boundaries)))
    
    def _split_large_section(self, text: str, offset: int, target_size: int) -> List[int]:
        """Split a large section into smaller chunks"""
        boundaries = []
        words = text.split()
        current_chunk = []
        current_pos = offset
        
        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= target_size:
                # Find a good break point
                chunk_text = ' '.join(current_chunk)
                break_point = self._find_sentence_boundary(chunk_text)
                if break_point > 0:
                    boundaries.append(current_pos + break_point)
                    current_pos += break_point
                    # Reset for next chunk
                    remaining_text = chunk_text[break_point:].strip()
                    current_chunk = remaining_text.split() if remaining_text else []
                else:
                    # Force break at target size
                    boundaries.append(current_pos + len(chunk_text))
                    current_pos += len(chunk_text)
                    current_chunk = []
        
        return boundaries
    
    def _find_sentence_boundary(self, text: str) -> int:
        """Find the best sentence boundary within text"""
        # Look for sentence endings near the end of the text
        sentence_endings = list(re.finditer(r'[.!?]+\s+', text))
        if sentence_endings:
            # Return the position after the last sentence ending
            return sentence_endings[-1].end()
        return 0
    
    def _create_context_chunks(self, text: str, boundaries: List[int], 
                             content_analysis: Dict) -> List[ContextChunk]:
        """Create context-aware chunks with full analysis"""
        chunks = []
        
        for i in range(len(boundaries) - 1):
            start_pos = boundaries[i]
            end_pos = boundaries[i + 1]
            chunk_text = text[start_pos:end_pos].strip()
            
            if not chunk_text:
                continue
            
            # Calculate context windows
            context_start = max(0, start_pos - self.strategy.context_window * 4)  # Rough char estimate
            context_end = min(len(text), end_pos + self.strategy.context_window * 4)
            
            preceding_context = text[context_start:start_pos].strip()
            following_context = text[end_pos:context_end].strip()
            
            # Analyze chunk content
            chunk_analysis = self._analyze_chunk_content(chunk_text, content_analysis)
            
            # Calculate metrics
            word_count = len(chunk_text.split())
            sentence_count = len(re.split(r'[.!?]+', chunk_text))
            token_count = self._estimate_token_count(chunk_text)
            
            # Create chunk
            chunk = ContextChunk(
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos,
                chunk_id=f"chunk_{i:03d}",
                level="paragraph",  # Default level
                
                # Context
                preceding_context=preceding_context,
                following_context=following_context,
                narrative_position=start_pos / len(text),
                
                # Semantic properties
                semantic_density=chunk_analysis['semantic_density'],
                coherence_score=chunk_analysis['coherence_score'],
                topic_continuity=chunk_analysis['topic_continuity'],
                
                # Structural properties
                sentence_count=sentence_count,
                word_count=word_count,
                token_count=token_count,
                
                # Content analysis
                content_type=chunk_analysis['content_type'],
                themes=chunk_analysis['themes'],
                entities=chunk_analysis['entities'],
                key_concepts=chunk_analysis['key_concepts'],
                
                # Quality metrics
                readability_score=chunk_analysis['readability_score'],
                information_density=chunk_analysis['information_density'],
                dialogue_potential=chunk_analysis['dialogue_potential'],
                
                # Relationships (to be filled later)
                related_chunks=[],
                dependency_chunks=[]
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _analyze_chunk_content(self, text: str, global_analysis: Dict) -> Dict[str, Any]:
        """Analyze individual chunk content"""
        analysis = {
            'semantic_density': 0.0,
            'coherence_score': 0.0,
            'topic_continuity': 0.0,
            'content_type': 'narrative',
            'themes': [],
            'entities': [],
            'key_concepts': [],
            'readability_score': 0.0,
            'information_density': 0.0,
            'dialogue_potential': 0.0
        }
        
        # Determine content type
        content_scores = {}
        for content_type, patterns in self.content_patterns.items():
            score = 0
            for pattern in patterns:
                score += len(re.findall(pattern, text, re.IGNORECASE))
            content_scores[content_type] = score
        
        analysis['content_type'] = max(content_scores, key=content_scores.get) if content_scores else 'narrative'
        
        # Calculate dialogue potential
        dialogue_indicators = content_scores.get('dialogue', 0)
        analysis['dialogue_potential'] = min(dialogue_indicators / max(len(text.split()) / 50, 1), 1.0)
        
        # Use semantic engine for detailed analysis
        if hasattr(self.semantic_engine, '_extract_entities'):
            analysis['entities'] = [e['text'] for e in self.semantic_engine._extract_entities(text)]
            analysis['key_concepts'] = self.semantic_engine._extract_concepts(text)
            analysis['themes'] = self.semantic_engine._identify_themes(text)
        
        # Calculate semantic density
        words = len(text.split())
        entities = len(analysis['entities'])
        concepts = len(analysis['key_concepts'])
        analysis['semantic_density'] = (entities + concepts) / max(words / 10, 1)
        analysis['semantic_density'] = min(analysis['semantic_density'], 1.0)
        
        # Calculate information density
        unique_words = len(set(text.lower().split()))
        analysis['information_density'] = unique_words / max(words, 1)
        
        # Simple readability score (based on sentence length and word complexity)
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = words / max(len(sentences), 1)
        long_words = len([w for w in text.split() if len(w) > 6])
        analysis['readability_score'] = max(0, 1.0 - (avg_sentence_length / 30) - (long_words / words))
        
        # Coherence score (simplified)
        analysis['coherence_score'] = min(
            analysis['semantic_density'] * 0.4 +
            analysis['information_density'] * 0.3 +
            analysis['readability_score'] * 0.3,
            1.0
        )
        
        return analysis
    
    def _optimize_chunk_relationships(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Analyze and optimize relationships between chunks"""
        if not chunks or not self.semantic_engine.sentence_model:
            return chunks
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk.text for chunk in chunks]
        try:
            embeddings = self.semantic_engine.sentence_model.encode(chunk_texts)
        except Exception as e:
            logger.warning(f"Could not generate embeddings for relationship analysis: {e}")
            return chunks
        
        # Calculate similarities and relationships
        for i, chunk in enumerate(chunks):
            related_chunks = []
            dependency_chunks = []
            
            for j, other_chunk in enumerate(chunks):
                if i == j:
                    continue
                
                # Calculate semantic similarity
                similarity = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                
                # High similarity indicates related content
                if similarity > 0.7:
                    related_chunks.append(other_chunk.chunk_id)
                
                # Check for dependencies (references to entities/concepts from previous chunks)
                if j < i:  # Only look at previous chunks for dependencies
                    shared_entities = set(chunk.entities) & set(other_chunk.entities)
                    shared_concepts = set(chunk.key_concepts) & set(other_chunk.key_concepts)
                    
                    if shared_entities or shared_concepts:
                        dependency_chunks.append(other_chunk.chunk_id)
            
            chunk.related_chunks = related_chunks
            chunk.dependency_chunks = dependency_chunks
        
        return chunks
    
    def _validate_and_adjust_chunks(self, chunks: List[ContextChunk], original_text: str) -> List[ContextChunk]:
        """Validate chunks and make adjustments if needed"""
        validated_chunks = []
        
        for chunk in chunks:
            # Check size constraints
            if chunk.word_count < self.strategy.min_size:
                # Try to merge with next chunk
                chunk = self._try_merge_chunk(chunk, chunks, validated_chunks)
            elif chunk.word_count > self.strategy.max_size:
                # Split large chunk
                split_chunks = self._split_chunk(chunk)
                validated_chunks.extend(split_chunks)
                continue
            
            # Check quality constraints
            if (chunk.coherence_score < self.strategy.min_coherence or
                chunk.semantic_density < self.strategy.min_semantic_density):
                # Try to improve chunk by adjusting boundaries
                chunk = self._improve_chunk_quality(chunk, original_text)
            
            # Validate token budget
            if chunk.token_count > self.strategy.token_budget - self.strategy.prompt_tokens:
                # Adjust chunk size to fit token budget
                chunk = self._adjust_for_token_budget(chunk)
            
            validated_chunks.append(chunk)
        
        return validated_chunks
    
    def _try_merge_chunk(self, chunk: ContextChunk, all_chunks: List[ContextChunk], 
                        validated_chunks: List[ContextChunk]) -> ContextChunk:
        """Try to merge a small chunk with adjacent chunks"""
        # For now, just return the chunk as-is
        # In a full implementation, we would merge with the next chunk
        return chunk
    
    def _split_chunk(self, chunk: ContextChunk) -> List[ContextChunk]:
        """Split a large chunk into smaller ones"""
        # Simple split at the middle sentence
        sentences = re.split(r'[.!?]+', chunk.text)
        mid_point = len(sentences) // 2
        
        first_half = '. '.join(sentences[:mid_point]) + '.'
        second_half = '. '.join(sentences[mid_point:])
        
        # Create two new chunks (simplified)
        chunk1 = ContextChunk(
            text=first_half,
            start_pos=chunk.start_pos,
            end_pos=chunk.start_pos + len(first_half),
            chunk_id=chunk.chunk_id + "_a",
            level=chunk.level,
            preceding_context=chunk.preceding_context,
            following_context=second_half[:100],  # Preview of next chunk
            narrative_position=chunk.narrative_position,
            semantic_density=chunk.semantic_density,
            coherence_score=chunk.coherence_score,
            topic_continuity=chunk.topic_continuity,
            sentence_count=mid_point,
            word_count=len(first_half.split()),
            token_count=self._estimate_token_count(first_half),
            content_type=chunk.content_type,
            themes=chunk.themes,
            entities=chunk.entities[:len(chunk.entities)//2],
            key_concepts=chunk.key_concepts[:len(chunk.key_concepts)//2],
            readability_score=chunk.readability_score,
            information_density=chunk.information_density,
            dialogue_potential=chunk.dialogue_potential,
            related_chunks=[],
            dependency_chunks=chunk.dependency_chunks
        )
        
        chunk2 = ContextChunk(
            text=second_half,
            start_pos=chunk.start_pos + len(first_half),
            end_pos=chunk.end_pos,
            chunk_id=chunk.chunk_id + "_b",
            level=chunk.level,
            preceding_context=first_half[-100:],  # Preview of previous chunk
            following_context=chunk.following_context,
            narrative_position=chunk.narrative_position,
            semantic_density=chunk.semantic_density,
            coherence_score=chunk.coherence_score,
            topic_continuity=chunk.topic_continuity,
            sentence_count=len(sentences) - mid_point,
            word_count=len(second_half.split()),
            token_count=self._estimate_token_count(second_half),
            content_type=chunk.content_type,
            themes=chunk.themes,
            entities=chunk.entities[len(chunk.entities)//2:],
            key_concepts=chunk.key_concepts[len(chunk.key_concepts)//2:],
            readability_score=chunk.readability_score,
            information_density=chunk.information_density,
            dialogue_potential=chunk.dialogue_potential,
            related_chunks=[],
            dependency_chunks=[chunk1.chunk_id]
        )
        
        return [chunk1, chunk2]
    
    def _improve_chunk_quality(self, chunk: ContextChunk, original_text: str) -> ContextChunk:
        """Try to improve chunk quality by adjusting boundaries"""
        # For now, return as-is
        # In a full implementation, we would adjust boundaries to improve quality
        return chunk
    
    def _adjust_for_token_budget(self, chunk: ContextChunk) -> ContextChunk:
        """Adjust chunk size to fit within token budget"""
        available_tokens = self.strategy.token_budget - self.strategy.prompt_tokens
        
        if chunk.token_count <= available_tokens:
            return chunk
        
        # Estimate how much text to keep
        ratio = available_tokens / chunk.token_count
        target_words = int(chunk.word_count * ratio * 0.9)  # 10% safety margin
        
        words = chunk.text.split()
        if target_words < len(words):
            # Truncate at sentence boundary
            truncated_text = ' '.join(words[:target_words])
            last_sentence_end = max(
                truncated_text.rfind('.'),
                truncated_text.rfind('!'),
                truncated_text.rfind('?')
            )
            
            if last_sentence_end > 0:
                truncated_text = truncated_text[:last_sentence_end + 1]
            
            # Update chunk properties
            chunk.text = truncated_text
            chunk.word_count = len(truncated_text.split())
            chunk.token_count = self._estimate_token_count(truncated_text)
            chunk.end_pos = chunk.start_pos + len(truncated_text)
        
        return chunk
    
    def _estimate_word_count(self, text: str) -> int:
        """Estimate word count in text"""
        return len(text.split())
    
    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count for AI processing"""
        # Rough estimation: 1 token ≈ 0.75 words
        words = len(text.split())
        return int(words / 0.75)
    
    def get_chunk_summary(self, chunks: List[ContextChunk]) -> Dict[str, Any]:
        """Generate a summary of the chunking results"""
        if not chunks:
            return {}
        
        total_words = sum(chunk.word_count for chunk in chunks)
        total_tokens = sum(chunk.token_count for chunk in chunks)
        
        content_types = Counter(chunk.content_type for chunk in chunks)
        avg_coherence = np.mean([chunk.coherence_score for chunk in chunks])
        avg_semantic_density = np.mean([chunk.semantic_density for chunk in chunks])
        
        return {
            'total_chunks': len(chunks),
            'total_words': total_words,
            'total_tokens': total_tokens,
            'avg_words_per_chunk': total_words / len(chunks),
            'avg_tokens_per_chunk': total_tokens / len(chunks),
            'content_type_distribution': dict(content_types),
            'avg_coherence_score': avg_coherence,
            'avg_semantic_density': avg_semantic_density,
            'chunks_with_dialogue': sum(1 for chunk in chunks if chunk.dialogue_potential > 0.3),
            'chunks_with_high_density': sum(1 for chunk in chunks if chunk.semantic_density > 0.6),
            'narrative_coverage': {
                'beginning': sum(1 for chunk in chunks if chunk.narrative_position < 0.33),
                'middle': sum(1 for chunk in chunks if 0.33 <= chunk.narrative_position < 0.67),
                'end': sum(1 for chunk in chunks if chunk.narrative_position >= 0.67)
            }
        }
    
    def export_chunks_for_ai(self, chunks: List[ContextChunk], 
                           include_context: bool = True) -> List[Dict[str, Any]]:
        """Export chunks in format suitable for AI processing"""
        exported_chunks = []
        
        for chunk in chunks:
            chunk_data = {
                'id': chunk.chunk_id,
                'text': chunk.text,
                'word_count': chunk.word_count,
                'token_count': chunk.token_count,
                'content_type': chunk.content_type,
                'narrative_position': chunk.narrative_position,
                'quality_scores': {
                    'semantic_density': chunk.semantic_density,
                    'coherence_score': chunk.coherence_score,
                    'readability_score': chunk.readability_score,
                    'dialogue_potential': chunk.dialogue_potential
                },
                'themes': chunk.themes,
                'entities': chunk.entities,
                'key_concepts': chunk.key_concepts
            }
            
            if include_context:
                chunk_data['context'] = {
                    'preceding': chunk.preceding_context,
                    'following': chunk.following_context,
                    'related_chunks': chunk.related_chunks,
                    'dependency_chunks': chunk.dependency_chunks
                }
            
            exported_chunks.append(chunk_data)
        
        return exported_chunks


def test_context_aware_chunker():
    """Test the context-aware chunking engine"""
    print("🧪 Testing Context-Aware Chunking Engine...")
    
    # Test text with various content types
    test_text = """
    Chapter 1: The Beginning
    
    It was a dark and stormy night when Sarah first discovered her unusual ability. 
    She had always been different, but this was something extraordinary. The rain 
    pounded against her window as she sat in her room, contemplating the strange 
    events of the day.
    
    "I can't believe this is happening," she whispered to herself, staring at her 
    hands in amazement. The power seemed to flow through her fingers like electricity.
    
    However, Sarah didn't understand the full implications of her discovery. According 
    to ancient texts, such abilities were rare and often came with great responsibility. 
    The old legends spoke of chosen ones who could manipulate the very fabric of reality.
    
    Meanwhile, across town, Dr. Marcus Chen was conducting his own research into 
    paranormal phenomena. His laboratory was filled with sophisticated equipment 
    designed to detect unusual energy patterns. For years, he had been searching 
    for proof that such abilities existed.
    
    The connection between Sarah and Dr. Chen would soon become apparent. Their 
    destinies were intertwined in ways neither could imagine. The storm that night 
    was just the beginning of a much larger adventure.
    """
    
    # Initialize chunker with custom strategy
    strategy = ChunkingStrategy(
        target_size=150,  # Smaller chunks for testing
        min_size=50,
        max_size=250,
        token_budget=500,
        preserve_dialogue=True,
        adaptive_sizing=True
    )
    
    chunker = ContextAwareChunker(strategy)
    
    # Perform chunking
    chunks = chunker.chunk_with_context(test_text)
    
    # Display results
    print(f"\n📊 Chunking Results:")
    print(f"   Total chunks: {len(chunks)}")
    
    summary = chunker.get_chunk_summary(chunks)
    print(f"   Average words per chunk: {summary['avg_words_per_chunk']:.1f}")
    print(f"   Average coherence score: {summary['avg_coherence_score']:.3f}")
    print(f"   Content type distribution: {summary['content_type_distribution']}")
    print(f"   Chunks with dialogue: {summary['chunks_with_dialogue']}")
    
    # Show detailed chunk analysis
    for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
        print(f"\n📝 Chunk {i+1} ({chunk.chunk_id}):")
        print(f"   Content type: {chunk.content_type}")
        print(f"   Word count: {chunk.word_count}")
        print(f"   Semantic density: {chunk.semantic_density:.3f}")
        print(f"   Dialogue potential: {chunk.dialogue_potential:.3f}")
        print(f"   Themes: {chunk.themes}")
        print(f"   Entities: {chunk.entities}")
        print(f"   Text preview: {chunk.text[:100]}...")
        if chunk.related_chunks:
            print(f"   Related chunks: {chunk.related_chunks}")
    
    # Test export functionality
    exported = chunker.export_chunks_for_ai(chunks, include_context=True)
    print(f"\n📤 Export test: {len(exported)} chunks exported for AI processing")
    
    print("\n✅ Context-aware chunking engine test completed!")

if __name__ == "__main__":
    test_context_aware_chunker()

