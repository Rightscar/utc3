"""
spaCy Theme-Based Content Discovery Module
Intelligent extraction of thematically relevant content chunks using spaCy NLP
"""

import os
import re
import logging
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import streamlit as st

# spaCy and NLP dependencies
try:
    import spacy
    from spacy.matcher import PhraseMatcher, Matcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# spaCy types (needed for type hints)
try:
    from spacy.tokens import Doc, Span
    SPACY_TYPES_AVAILABLE = True
except ImportError:
    SPACY_TYPES_AVAILABLE = False
    # Create dummy classes for type hints when spaCy is not available
    class Doc:
        pass
    class Span:
        pass

# Alternative NLP libraries
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

@dataclass
class ThemeMatch:
    """Represents a thematic match found in content"""
    keyword: str
    matched_text: str
    context_before: str
    context_after: str
    full_chunk: str
    start_pos: int
    end_pos: int
    confidence: float
    sentence_index: int
    paragraph_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ThemeDiscoveryConfig:
    """Configuration for theme discovery"""
    context_sentences_before: int = 2
    context_sentences_after: int = 2
    min_chunk_length: int = 50
    max_chunk_length: int = 2000
    similarity_threshold: float = 0.7
    case_sensitive: bool = False
    include_synonyms: bool = True
    extract_entities: bool = True
    language: str = "en"

class SpacyThemeDiscovery:
    """Advanced theme-based content discovery using spaCy"""
    
    def __init__(self, config: Optional[ThemeDiscoveryConfig] = None):
        self.config = config or ThemeDiscoveryConfig()
        self.logger = logging.getLogger(__name__)
        self.nlp = None
        self.phrase_matcher = None
        self.matcher = None
        self.sentence_model = None
        self.spacy_available = SPACY_AVAILABLE  # Initialize from global
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize spaCy and other NLP models"""
        try:
            if self.spacy_available:
                # Try to load spaCy model
                model_name = f"{self.config.language}_core_web_sm"
                try:
                    self.nlp = spacy.load(model_name)
                    self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
                    self.matcher = Matcher(self.nlp.vocab)
                    self.logger.info(f"Loaded spaCy model: {model_name}")
                except OSError:
                    # Fallback to basic model
                    try:
                        self.nlp = spacy.load("en_core_web_sm")
                        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
                        self.matcher = Matcher(self.nlp.vocab)
                        self.logger.info("Loaded fallback spaCy model: en_core_web_sm")
                    except OSError:
                        self.logger.warning("No spaCy model available. Using fallback methods.")
                        # Don't modify global SPACY_AVAILABLE, just set instance variable
                        self.spacy_available = False
            
            # Initialize sentence transformer for semantic similarity
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.logger.info("Loaded sentence transformer model")
                except Exception as e:
                    self.logger.warning(f"Failed to load sentence transformer: {e}")
                    
        except Exception as e:
            self.logger.error(f"Model initialization error: {e}")
    
    def discover_themes(
        self, 
        content: str, 
        themes: List[str],
        custom_patterns: Optional[List[Dict]] = None
    ) -> List[ThemeMatch]:
        """
        Discover thematic content chunks based on keywords and patterns
        
        Args:
            content: Full text content to analyze
            themes: List of theme keywords/phrases
            custom_patterns: Optional custom spaCy patterns
            
        Returns:
            List of ThemeMatch objects
        """
        if not content or not themes:
            return []
        
        # Choose processing method based on available libraries
        if self.spacy_available and self.nlp:
            return self._discover_with_spacy(content, themes, custom_patterns)
        elif NLTK_AVAILABLE:
            return self._discover_with_nltk(content, themes)
        else:
            return self._discover_with_regex(content, themes)
    
    def _discover_with_spacy(
        self, 
        content: str, 
        themes: List[str],
        custom_patterns: Optional[List[Dict]] = None
    ) -> List[ThemeMatch]:
        """Advanced theme discovery using spaCy"""
        try:
            # Process the content
            doc = self.nlp(content)
            
            # Prepare phrase patterns
            patterns = []
            theme_to_pattern = {}
            
            for theme in themes:
                theme_doc = self.nlp(theme.lower())
                patterns.append(theme_doc)
                theme_to_pattern[theme.lower()] = theme
            
            # Add patterns to matcher
            self.phrase_matcher.clear()
            self.phrase_matcher.add("THEMES", patterns)
            
            # Add custom patterns if provided
            if custom_patterns:
                self.matcher.clear()
                for i, pattern in enumerate(custom_patterns):
                    self.matcher.add(f"CUSTOM_{i}", [pattern])
            
            # Find matches
            matches = []
            phrase_matches = self.phrase_matcher(doc)
            
            # Process phrase matches
            for match_id, start, end in phrase_matches:
                span = doc[start:end]
                original_theme = theme_to_pattern.get(span.text.lower(), span.text)
                
                theme_match = self._create_theme_match(
                    doc, span, original_theme, start, end
                )
                if theme_match:
                    matches.append(theme_match)
            
            # Process custom pattern matches
            if custom_patterns:
                custom_matches = self.matcher(doc)
                for match_id, start, end in custom_matches:
                    span = doc[start:end]
                    theme_match = self._create_theme_match(
                        doc, span, f"custom_pattern_{match_id}", start, end
                    )
                    if theme_match:
                        matches.append(theme_match)
            
            # Remove duplicates and sort by position
            unique_matches = self._deduplicate_matches(matches)
            return sorted(unique_matches, key=lambda x: x.start_pos)
            
        except Exception as e:
            self.logger.error(f"spaCy theme discovery error: {e}")
            return self._discover_with_nltk(content, themes)
    
    def _create_theme_match(
        self, 
        doc: Doc, 
        span: Span, 
        theme: str, 
        start: int, 
        end: int
    ) -> Optional[ThemeMatch]:
        """Create a ThemeMatch object from spaCy span"""
        try:
            # Get sentence and paragraph indices
            sent_idx = 0
            para_idx = 0
            
            for i, sent in enumerate(doc.sents):
                if span.start >= sent.start and span.end <= sent.end:
                    sent_idx = i
                    break
            
            # Count paragraphs (rough approximation)
            text_before_span = doc[:span.start].text
            para_idx = text_before_span.count('\n\n')
            
            # Extract context
            context_before, context_after, full_chunk = self._extract_context_spacy(
                doc, span, sent_idx
            )
            
            # Calculate confidence (basic implementation)
            confidence = self._calculate_confidence_spacy(span, theme)
            
            # Extract entities and metadata
            metadata = self._extract_metadata_spacy(doc, span)
            
            return ThemeMatch(
                keyword=theme,
                matched_text=span.text,
                context_before=context_before,
                context_after=context_after,
                full_chunk=full_chunk,
                start_pos=span.start_char,
                end_pos=span.end_char,
                confidence=confidence,
                sentence_index=sent_idx,
                paragraph_index=para_idx,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error creating theme match: {e}")
            return None
    
    def _extract_context_spacy(
        self, 
        doc: Doc, 
        span: Span, 
        sent_idx: int
    ) -> Tuple[str, str, str]:
        """Extract context around the matched span"""
        sentences = list(doc.sents)
        
        # Get context sentences
        start_sent = max(0, sent_idx - self.config.context_sentences_before)
        end_sent = min(len(sentences), sent_idx + self.config.context_sentences_after + 1)
        
        context_sentences = sentences[start_sent:end_sent]
        
        # Find the sentence containing the match
        match_sent = sentences[sent_idx]
        
        # Extract before and after context
        context_before = " ".join([sent.text for sent in sentences[start_sent:sent_idx]])
        context_after = " ".join([sent.text for sent in sentences[sent_idx+1:end_sent]])
        
        # Full chunk
        full_chunk = " ".join([sent.text for sent in context_sentences])
        
        # Ensure chunk length limits
        if len(full_chunk) > self.config.max_chunk_length:
            # Truncate while preserving the match
            match_start = full_chunk.find(span.text)
            if match_start != -1:
                start_pos = max(0, match_start - self.config.max_chunk_length // 2)
                end_pos = min(len(full_chunk), match_start + len(span.text) + self.config.max_chunk_length // 2)
                full_chunk = full_chunk[start_pos:end_pos]
                
                # Update context accordingly
                if start_pos > 0:
                    context_before = full_chunk[:match_start - start_pos]
                if end_pos < len(full_chunk):
                    match_end = match_start + len(span.text) - start_pos
                    context_after = full_chunk[match_end:]
        
        return context_before.strip(), context_after.strip(), full_chunk.strip()
    
    def _calculate_confidence_spacy(self, span: Span, theme: str) -> float:
        """Calculate confidence score for the match"""
        # Basic confidence calculation
        confidence = 0.5  # Base confidence
        
        # Boost for exact matches
        if span.text.lower() == theme.lower():
            confidence += 0.3
        
        # Boost for entity matches
        if span.ent_type_:
            confidence += 0.2
        
        # Boost for important POS tags
        important_pos = {'NOUN', 'VERB', 'ADJ', 'PROPN'}
        if any(token.pos_ in important_pos for token in span):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _extract_metadata_spacy(self, doc: Doc, span: Span) -> Dict[str, Any]:
        """Extract metadata from the span"""
        metadata = {}
        
        # Entity information
        if span.ent_type_:
            metadata['entity_type'] = span.ent_type_
            metadata['entity_label'] = span.label_
        
        # POS tags
        metadata['pos_tags'] = [token.pos_ for token in span]
        
        # Dependency information
        metadata['dependencies'] = [token.dep_ for token in span]
        
        # Surrounding entities
        sent = span.sent
        entities = [(ent.text, ent.label_) for ent in sent.ents if ent != span]
        metadata['surrounding_entities'] = entities
        
        return metadata
    
    def _discover_with_nltk(self, content: str, themes: List[str]) -> List[ThemeMatch]:
        """Fallback theme discovery using NLTK"""
        try:
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            
            sentences = sent_tokenize(content)
            matches = []
            
            for theme in themes:
                theme_lower = theme.lower()
                
                for sent_idx, sentence in enumerate(sentences):
                    sentence_lower = sentence.lower()
                    
                    if theme_lower in sentence_lower:
                        # Find the position of the match
                        match_start = sentence_lower.find(theme_lower)
                        match_end = match_start + len(theme)
                        
                        # Extract context
                        context_before, context_after, full_chunk = self._extract_context_nltk(
                            sentences, sent_idx, sentence
                        )
                        
                        # Calculate paragraph index (rough)
                        para_idx = content[:content.find(sentence)].count('\n\n')
                        
                        theme_match = ThemeMatch(
                            keyword=theme,
                            matched_text=theme,
                            context_before=context_before,
                            context_after=context_after,
                            full_chunk=full_chunk,
                            start_pos=content.find(sentence) + match_start,
                            end_pos=content.find(sentence) + match_end,
                            confidence=0.7,  # Default confidence for NLTK
                            sentence_index=sent_idx,
                            paragraph_index=para_idx,
                            metadata={'method': 'nltk'}
                        )
                        matches.append(theme_match)
            
            return self._deduplicate_matches(matches)
            
        except Exception as e:
            self.logger.error(f"NLTK theme discovery error: {e}")
            return self._discover_with_regex(content, themes)
    
    def _extract_context_nltk(
        self, 
        sentences: List[str], 
        sent_idx: int, 
        match_sentence: str
    ) -> Tuple[str, str, str]:
        """Extract context using NLTK sentence tokenization"""
        start_idx = max(0, sent_idx - self.config.context_sentences_before)
        end_idx = min(len(sentences), sent_idx + self.config.context_sentences_after + 1)
        
        context_before = " ".join(sentences[start_idx:sent_idx])
        context_after = " ".join(sentences[sent_idx+1:end_idx])
        full_chunk = " ".join(sentences[start_idx:end_idx])
        
        return context_before.strip(), context_after.strip(), full_chunk.strip()
    
    def _discover_with_regex(self, content: str, themes: List[str]) -> List[ThemeMatch]:
        """Basic fallback using regex"""
        matches = []
        
        for theme in themes:
            # Create regex pattern
            if self.config.case_sensitive:
                pattern = re.compile(re.escape(theme))
            else:
                pattern = re.compile(re.escape(theme), re.IGNORECASE)
            
            for match in pattern.finditer(content):
                start_pos = match.start()
                end_pos = match.end()
                
                # Extract context (simple approach)
                context_start = max(0, start_pos - 200)
                context_end = min(len(content), end_pos + 200)
                
                full_chunk = content[context_start:context_end]
                context_before = content[context_start:start_pos]
                context_after = content[end_pos:context_end]
                
                # Rough sentence and paragraph indices
                sent_idx = content[:start_pos].count('.') + content[:start_pos].count('!') + content[:start_pos].count('?')
                para_idx = content[:start_pos].count('\n\n')
                
                theme_match = ThemeMatch(
                    keyword=theme,
                    matched_text=match.group(),
                    context_before=context_before.strip(),
                    context_after=context_after.strip(),
                    full_chunk=full_chunk.strip(),
                    start_pos=start_pos,
                    end_pos=end_pos,
                    confidence=0.5,  # Default confidence for regex
                    sentence_index=sent_idx,
                    paragraph_index=para_idx,
                    metadata={'method': 'regex'}
                )
                matches.append(theme_match)
        
        return self._deduplicate_matches(matches)
    
    def _deduplicate_matches(self, matches: List[ThemeMatch]) -> List[ThemeMatch]:
        """Remove duplicate or overlapping matches"""
        if not matches:
            return []
        
        # Sort by start position
        sorted_matches = sorted(matches, key=lambda x: x.start_pos)
        
        # Remove overlaps
        deduplicated = [sorted_matches[0]]
        
        for match in sorted_matches[1:]:
            last_match = deduplicated[-1]
            
            # Check for overlap
            if match.start_pos >= last_match.end_pos:
                deduplicated.append(match)
            elif match.confidence > last_match.confidence:
                # Replace with higher confidence match
                deduplicated[-1] = match
        
        return deduplicated
    
    def enhance_themes_with_similarity(
        self, 
        content: str, 
        themes: List[str],
        similarity_threshold: float = None
    ) -> List[ThemeMatch]:
        """Enhance theme discovery with semantic similarity"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE or not self.sentence_model:
            return self.discover_themes(content, themes)
        
        threshold = similarity_threshold or self.config.similarity_threshold
        
        try:
            # Get basic matches first
            basic_matches = self.discover_themes(content, themes)
            
            # Split content into sentences for similarity analysis
            if NLTK_AVAILABLE:
                sentences = sent_tokenize(content)
            else:
                sentences = content.split('.')
            
            # Encode themes and sentences
            theme_embeddings = self.sentence_model.encode(themes)
            sentence_embeddings = self.sentence_model.encode(sentences)
            
            # Find semantically similar sentences
            similarity_matches = []
            
            for i, sentence in enumerate(sentences):
                for j, theme in enumerate(themes):
                    similarity = cosine_similarity(
                        [sentence_embeddings[i]], 
                        [theme_embeddings[j]]
                    )[0][0]
                    
                    if similarity >= threshold:
                        # Create a similarity-based match
                        start_pos = content.find(sentence)
                        if start_pos != -1:
                            theme_match = ThemeMatch(
                                keyword=theme,
                                matched_text=sentence[:50] + "..." if len(sentence) > 50 else sentence,
                                context_before="",
                                context_after="",
                                full_chunk=sentence,
                                start_pos=start_pos,
                                end_pos=start_pos + len(sentence),
                                confidence=similarity,
                                sentence_index=i,
                                paragraph_index=content[:start_pos].count('\n\n'),
                                metadata={'method': 'semantic_similarity', 'similarity_score': similarity}
                            )
                            similarity_matches.append(theme_match)
            
            # Combine and deduplicate
            all_matches = basic_matches + similarity_matches
            return self._deduplicate_matches(all_matches)
            
        except Exception as e:
            self.logger.error(f"Similarity enhancement error: {e}")
            return basic_matches
    
    def get_theme_statistics(self, matches: List[ThemeMatch]) -> Dict[str, Any]:
        """Generate statistics about theme matches"""
        if not matches:
            return {}
        
        stats = {
            'total_matches': len(matches),
            'unique_themes': len(set(match.keyword for match in matches)),
            'average_confidence': sum(match.confidence for match in matches) / len(matches),
            'theme_counts': defaultdict(int),
            'confidence_distribution': {
                'high': 0,  # > 0.8
                'medium': 0,  # 0.5 - 0.8
                'low': 0  # < 0.5
            },
            'methods_used': set()
        }
        
        for match in matches:
            stats['theme_counts'][match.keyword] += 1
            
            if match.confidence > 0.8:
                stats['confidence_distribution']['high'] += 1
            elif match.confidence > 0.5:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
            
            if 'method' in match.metadata:
                stats['methods_used'].add(match.metadata['method'])
        
        return dict(stats)

# Streamlit UI Components
def create_theme_discovery_ui():
    """Create Streamlit UI for theme discovery"""
    st.subheader("🔍 Theme-Based Content Discovery")
    
    # Configuration section
    with st.expander("⚙️ Discovery Settings", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            context_before = st.slider("Context sentences before", 1, 5, 2)
            context_after = st.slider("Context sentences after", 1, 5, 2)
            similarity_threshold = st.slider("Similarity threshold", 0.5, 1.0, 0.7, 0.05)
        
        with col2:
            case_sensitive = st.checkbox("Case sensitive matching", False)
            include_synonyms = st.checkbox("Include synonyms", True)
            extract_entities = st.checkbox("Extract entities", True)
    
    # Theme input
    st.write("**Define Themes/Keywords:**")
    theme_input_method = st.radio(
        "Input method:",
        ["Manual entry", "Upload file", "Predefined categories"],
        horizontal=True
    )
    
    themes = []
    
    if theme_input_method == "Manual entry":
        theme_text = st.text_area(
            "Enter themes (one per line):",
            placeholder="suffering\nmind\nchoice\ntruth\nwisdom\nconsciousness",
            height=100
        )
        themes = [theme.strip() for theme in theme_text.split('\n') if theme.strip()]
    
    elif theme_input_method == "Upload file":
        uploaded_file = st.file_uploader("Upload themes file", type=['txt'])
        if uploaded_file:
            themes = [line.decode('utf-8').strip() for line in uploaded_file.readlines()]
    
    elif theme_input_method == "Predefined categories":
        category = st.selectbox(
            "Select category:",
            ["Spiritual/Consciousness", "Psychology/Mind", "Philosophy", "Self-Development", "Custom"]
        )
        
        predefined_themes = {
            "Spiritual/Consciousness": [
                "consciousness", "awareness", "enlightenment", "awakening", "meditation",
                "mindfulness", "presence", "being", "soul", "spirit", "divine", "sacred"
            ],
            "Psychology/Mind": [
                "mind", "thought", "emotion", "feeling", "memory", "perception",
                "cognition", "behavior", "psychology", "mental", "subconscious"
            ],
            "Philosophy": [
                "truth", "reality", "existence", "meaning", "purpose", "ethics",
                "morality", "wisdom", "knowledge", "understanding", "philosophy"
            ],
            "Self-Development": [
                "growth", "development", "improvement", "learning", "skill",
                "habit", "goal", "success", "achievement", "potential", "transformation"
            ]
        }
        
        if category in predefined_themes:
            themes = predefined_themes[category]
            st.write(f"Selected themes: {', '.join(themes)}")
    
    # Display current themes
    if themes:
        st.write(f"**Active themes ({len(themes)}):** {', '.join(themes[:10])}")
        if len(themes) > 10:
            st.write(f"... and {len(themes) - 10} more")
    
    return themes, ThemeDiscoveryConfig(
        context_sentences_before=context_before,
        context_sentences_after=context_after,
        similarity_threshold=similarity_threshold,
        case_sensitive=case_sensitive,
        include_synonyms=include_synonyms,
        extract_entities=extract_entities
    )

def display_theme_matches(matches: List[ThemeMatch], content: str):
    """Display theme matches in Streamlit UI"""
    if not matches:
        st.info("No theme matches found. Try different keywords or adjust settings.")
        return []
    
    st.success(f"Found {len(matches)} theme matches!")
    
    # Statistics
    stats = SpacyThemeDiscovery().get_theme_statistics(matches)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", stats.get('total_matches', 0))
    with col2:
        st.metric("Unique Themes", stats.get('unique_themes', 0))
    with col3:
        st.metric("Avg Confidence", f"{stats.get('average_confidence', 0):.2f}")
    with col4:
        st.metric("High Confidence", stats.get('confidence_distribution', {}).get('high', 0))
    
    # Match selection and review
    approved_matches = []
    
    for i, match in enumerate(matches):
        with st.expander(f"Match {i+1}: '{match.keyword}' (Confidence: {match.confidence:.2f})", expanded=False):
            
            # Display match details
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Matched Text:**")
                st.code(match.matched_text)
                
                st.write("**Full Context:**")
                # Highlight the matched text in context
                highlighted_chunk = match.full_chunk.replace(
                    match.matched_text, 
                    f"**{match.matched_text}**"
                )
                st.markdown(highlighted_chunk)
            
            with col2:
                st.write("**Match Info:**")
                st.write(f"Position: {match.start_pos}-{match.end_pos}")
                st.write(f"Sentence: {match.sentence_index}")
                st.write(f"Paragraph: {match.paragraph_index}")
                
                if match.metadata:
                    st.write("**Metadata:**")
                    for key, value in match.metadata.items():
                        if isinstance(value, list) and len(value) > 3:
                            st.write(f"{key}: {value[:3]}...")
                        else:
                            st.write(f"{key}: {value}")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"✅ Approve", key=f"approve_{i}"):
                    approved_matches.append(match)
                    st.success("Match approved!")
            
            with col2:
                if st.button(f"✏️ Edit", key=f"edit_{i}"):
                    # Allow editing of the chunk
                    edited_chunk = st.text_area(
                        "Edit chunk:",
                        value=match.full_chunk,
                        key=f"edit_text_{i}",
                        height=100
                    )
                    
                    if st.button(f"Save Edit", key=f"save_edit_{i}"):
                        # Create new match with edited content
                        edited_match = ThemeMatch(
                            keyword=match.keyword,
                            matched_text=match.matched_text,
                            context_before=match.context_before,
                            context_after=match.context_after,
                            full_chunk=edited_chunk,
                            start_pos=match.start_pos,
                            end_pos=match.end_pos,
                            confidence=match.confidence,
                            sentence_index=match.sentence_index,
                            paragraph_index=match.paragraph_index,
                            metadata={**match.metadata, 'edited': True}
                        )
                        approved_matches.append(edited_match)
                        st.success("Edited match approved!")
            
            with col3:
                if st.button(f"❌ Skip", key=f"skip_{i}"):
                    st.info("Match skipped")
    
    return approved_matches

def check_nlp_dependencies():
    """Check availability of NLP dependencies"""
    status = {
        'spacy': SPACY_AVAILABLE,
        'nltk': NLTK_AVAILABLE,
        'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE
    }
    
    # Check for spaCy models
    if SPACY_AVAILABLE:
        try:
            import spacy
            spacy.load("en_core_web_sm")
            status['spacy_model'] = True
        except OSError:
            status['spacy_model'] = False
    
    return status

# Example usage and testing
def test_theme_discovery():
    """Test the theme discovery functionality"""
    sample_content = """
    The nature of consciousness has puzzled philosophers and scientists for centuries. 
    What is the mind, and how does it relate to the brain? These questions touch the 
    very core of human existence and our understanding of reality.
    
    In meditation, we often discover that the mind is not what we thought it was. 
    The constant stream of thoughts and emotions that we identify with begins to 
    reveal itself as something we can observe rather than something we are.
    
    Truth, it seems, is not something we can grasp with the intellect alone. 
    It requires a different kind of knowing, one that emerges from direct experience 
    rather than conceptual understanding.
    """
    
    themes = ["consciousness", "mind", "truth", "meditation", "reality"]
    
    discovery = SpacyThemeDiscovery()
    matches = discovery.discover_themes(sample_content, themes)
    
    print(f"Found {len(matches)} matches:")
    for match in matches:
        print(f"- {match.keyword}: {match.matched_text} (confidence: {match.confidence:.2f})")
    
    return matches

if __name__ == "__main__":
    # Run test
    test_matches = test_theme_discovery()
    print(f"Test completed with {len(test_matches)} matches")

