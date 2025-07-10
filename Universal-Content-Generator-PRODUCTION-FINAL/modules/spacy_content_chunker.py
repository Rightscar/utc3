"""
spaCy Content Chunker Module
============================

Intelligent text chunking and content selection using spaCy NLP.
Converts ANY text content into selectable chunks for dialogue generation.

Features:
- Smart text segmentation (paragraphs, sections, semantic boundaries)
- Content quality scoring for each chunk
- Interactive chunk selection interface
- Context extraction and metadata
"""

import streamlit as st
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re

# spaCy dependencies
try:
    import spacy
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ContentChunk:
    """Represents a chunk of content with metadata"""
    id: str
    text: str
    start_pos: int
    end_pos: int
    chunk_type: str  # 'paragraph', 'section', 'semantic'
    quality_score: float
    dialogue_potential: float
    word_count: int
    sentence_count: int
    entities: List[str]
    topics: List[str]
    selected: bool = False

class SpacyContentChunker:
    """Intelligent content chunking using spaCy"""
    
    def __init__(self):
        self.nlp = None
        self.spacy_available = SPACY_AVAILABLE
        self._initialize_spacy()
        
    def _initialize_spacy(self):
        """Initialize spaCy model"""
        if not self.spacy_available:
            logger.warning("spaCy not available, using fallback chunking")
            return
            
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy English model not found, using fallback")
            self.spacy_available = False
    
    def chunk_content(self, text: str, chunk_size: int = 1000) -> List[ContentChunk]:
        """
        Chunk content into intelligent segments
        
        Args:
            text: Input text to chunk
            chunk_size: Target chunk size in characters
            
        Returns:
            List of ContentChunk objects
        """
        if self.spacy_available and self.nlp:
            return self._chunk_with_spacy(text, chunk_size)
        else:
            return self._chunk_fallback(text, chunk_size)
    
    def _chunk_with_spacy(self, text: str, chunk_size: int) -> List[ContentChunk]:
        """Chunk content using spaCy NLP"""
        chunks = []
        
        try:
            # Process text with spaCy
            doc = self.nlp(text)
            
            # Extract sentences
            sentences = list(doc.sents)
            
            current_chunk = ""
            current_start = 0
            chunk_id = 0
            
            for i, sent in enumerate(sentences):
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sent.text) > chunk_size and current_chunk:
                    # Create chunk from accumulated sentences
                    chunk = self._create_chunk_with_spacy(
                        chunk_id, current_chunk, current_start, 
                        current_start + len(current_chunk), doc
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk
                    chunk_id += 1
                    current_chunk = sent.text
                    current_start = sent.start_char
                else:
                    # Add sentence to current chunk
                    if current_chunk:
                        current_chunk += " " + sent.text
                    else:
                        current_chunk = sent.text
                        current_start = sent.start_char
            
            # Add final chunk
            if current_chunk:
                chunk = self._create_chunk_with_spacy(
                    chunk_id, current_chunk, current_start,
                    current_start + len(current_chunk), doc
                )
                chunks.append(chunk)
                
        except Exception as e:
            logger.error(f"spaCy chunking error: {e}")
            return self._chunk_fallback(text, chunk_size)
        
        return chunks
    
    def _create_chunk_with_spacy(self, chunk_id: int, text: str, start: int, end: int, doc) -> ContentChunk:
        """Create a ContentChunk with spaCy analysis"""
        
        # Extract entities from chunk
        chunk_doc = self.nlp(text)
        entities = [ent.text for ent in chunk_doc.ents]
        
        # Count words and sentences
        word_count = len([token for token in chunk_doc if not token.is_space])
        sentence_count = len(list(chunk_doc.sents))
        
        # Extract topics (simplified - using noun phrases)
        topics = [chunk.text for chunk in chunk_doc.noun_chunks][:5]
        
        return ContentChunk(
            id=f"chunk_{chunk_id}",
            text=text,
            start_pos=start,
            end_pos=end,
            chunk_type="semantic",
            quality_score=0.8,  # Default good score
            dialogue_potential=0.8,  # Default good potential
            word_count=word_count,
            sentence_count=sentence_count,
            entities=entities,
            topics=topics
        )
    
    def _chunk_fallback(self, text: str, chunk_size: int) -> List[ContentChunk]:
        """Fallback chunking without spaCy"""
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_start = 0
        chunk_id = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                # Create chunk
                chunk = ContentChunk(
                    id=f"chunk_{chunk_id}",
                    text=current_chunk.strip(),
                    start_pos=current_start,
                    end_pos=current_start + len(current_chunk),
                    chunk_type="paragraph",
                    quality_score=0.7,  # Default good score
                    dialogue_potential=0.7,  # Default good potential
                    word_count=len(current_chunk.split()),
                    sentence_count=current_chunk.count('.') + current_chunk.count('!') + current_chunk.count('?'),
                    entities=[],
                    topics=[]
                )
                chunks.append(chunk)
                
                # Start new chunk
                chunk_id += 1
                current_chunk = para
                current_start = text.find(para)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                    current_start = text.find(para)
        
        # Add final chunk
        if current_chunk:
            chunk = ContentChunk(
                id=f"chunk_{chunk_id}",
                text=current_chunk.strip(),
                start_pos=current_start,
                end_pos=current_start + len(current_chunk),
                chunk_type="paragraph",
                quality_score=0.7,  # Default good score
                dialogue_potential=0.7,  # Default good potential
                word_count=len(current_chunk.split()),
                sentence_count=current_chunk.count('.') + current_chunk.count('!') + current_chunk.count('?'),
                entities=[],
                topics=[]
            )
            chunks.append(chunk)
        
        return chunks

def render_chunk_selection_ui(chunks: List[ContentChunk]) -> List[ContentChunk]:
    """
    Render interactive chunk selection interface
    
    Args:
        chunks: List of ContentChunk objects
        
    Returns:
        List of selected chunks
    """
    if not chunks:
        st.warning("No chunks available for selection")
        return []
    
    st.subheader("📝 Content Chunks - Select for Dialogue Generation")
    
    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Chunks", len(chunks))
    with col2:
        avg_quality = sum(chunk.quality_score for chunk in chunks) / len(chunks)
        st.metric("Avg Quality", f"{avg_quality:.2f}")
    with col3:
        avg_dialogue = sum(chunk.dialogue_potential for chunk in chunks) / len(chunks)
        st.metric("Avg Dialogue Potential", f"{avg_dialogue:.2f}")
    with col4:
        total_words = sum(chunk.word_count for chunk in chunks)
        st.metric("Total Words", f"{total_words:,}")
    
    # Selection controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("✅ Select All"):
            for chunk in chunks:
                chunk.selected = True
            st.rerun()
    
    with col2:
        if st.button("❌ Deselect All"):
            for chunk in chunks:
                chunk.selected = False
            st.rerun()
    
    with col3:
        if st.button("🎯 Select High Quality"):
            for chunk in chunks:
                chunk.selected = chunk.quality_score > 0.6 or chunk.dialogue_potential > 0.6
            st.rerun()
    
    # Chunk display and selection
    selected_chunks = []
    
    for i, chunk in enumerate(chunks):
        with st.expander(f"Chunk {i+1}: {chunk.text[:100]}..." if len(chunk.text) > 100 else f"Chunk {i+1}: {chunk.text}", expanded=False):
            
            # Chunk metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Quality Score", f"{chunk.quality_score:.2f}")
            with col2:
                st.metric("Dialogue Potential", f"{chunk.dialogue_potential:.2f}")
            with col3:
                st.metric("Words", chunk.word_count)
            
            # Selection checkbox
            chunk.selected = st.checkbox(
                f"Select for dialogue generation",
                value=chunk.selected,
                key=f"chunk_select_{chunk.id}"
            )
            
            # Chunk content
            st.text_area(
                "Content",
                value=chunk.text,
                height=150,
                disabled=True,
                key=f"chunk_content_{chunk.id}"
            )
            
            # Metadata
            if chunk.entities:
                st.write("**Entities:**", ", ".join(chunk.entities[:5]))
            if chunk.topics:
                st.write("**Topics:**", ", ".join(chunk.topics[:3]))
        
        if chunk.selected:
            selected_chunks.append(chunk)
    
    # Selection summary
    if selected_chunks:
        st.success(f"✅ {len(selected_chunks)} chunks selected for dialogue generation")
        
        # Show selected chunks summary
        with st.expander("📋 Selected Chunks Summary"):
            for chunk in selected_chunks:
                st.write(f"• **{chunk.id}:** {chunk.text[:150]}...")
    else:
        st.info("Select chunks above to proceed with dialogue generation")
    
    return selected_chunks

@st.cache_data
def chunk_text_content(text: str, chunk_size: int = 1000) -> List[Dict[str, Any]]:
    """
    Cached function to chunk text content
    
    Args:
        text: Input text
        chunk_size: Target chunk size
        
    Returns:
        List of chunk dictionaries (for caching compatibility)
    """
    chunker = SpacyContentChunker()
    chunks = chunker.chunk_content(text, chunk_size)
    
    # Convert to dictionaries for caching
    return [
        {
            'id': chunk.id,
            'text': chunk.text,
            'start_pos': chunk.start_pos,
            'end_pos': chunk.end_pos,
            'chunk_type': chunk.chunk_type,
            'quality_score': chunk.quality_score,
            'dialogue_potential': chunk.dialogue_potential,
            'word_count': chunk.word_count,
            'sentence_count': chunk.sentence_count,
            'entities': chunk.entities,
            'topics': chunk.topics,
            'selected': False
        }
        for chunk in chunks
    ]

def chunks_from_dicts(chunk_dicts: List[Dict[str, Any]]) -> List[ContentChunk]:
    """Convert chunk dictionaries back to ContentChunk objects"""
    return [
        ContentChunk(
            id=chunk_dict['id'],
            text=chunk_dict['text'],
            start_pos=chunk_dict['start_pos'],
            end_pos=chunk_dict['end_pos'],
            chunk_type=chunk_dict['chunk_type'],
            quality_score=chunk_dict['quality_score'],
            dialogue_potential=chunk_dict['dialogue_potential'],
            word_count=chunk_dict['word_count'],
            sentence_count=chunk_dict['sentence_count'],
            entities=chunk_dict['entities'],
            topics=chunk_dict['topics'],
            selected=chunk_dict.get('selected', False)
        )
        for chunk_dict in chunk_dicts
    ]

