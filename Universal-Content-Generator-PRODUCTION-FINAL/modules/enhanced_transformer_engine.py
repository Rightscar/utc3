"""
Enhanced Transformer Engine
==========================

This module integrates advanced transformer models (BERT, RoBERTa, DistilBERT) 
with our existing spaCy-based pipeline to provide state-of-the-art semantic 
understanding and contextual processing capabilities.

Features:
- Multi-model transformer support (BERT, RoBERTa, DistilBERT)
- Contextual embeddings with attention mechanisms
- Advanced semantic similarity and clustering
- Efficient model caching and optimization
- Fallback to sentence transformers for compatibility
- Performance monitoring and benchmarking
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import time
from pathlib import Path
import pickle

# Core dependencies
try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        BertModel, BertTokenizer,
        RobertaModel, RobertaTokenizer,
        DistilBertModel, DistilBertTokenizer
    )
    import torch
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("⚠️ Transformers library not available, falling back to sentence transformers")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("⚠️ Sentence transformers not available")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import spacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TransformerConfig:
    """Configuration for transformer models"""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    batch_size: int = 16
    device: str = "auto"  # auto, cpu, cuda
    cache_embeddings: bool = True
    use_attention_weights: bool = True
    pooling_strategy: str = "mean"  # mean, max, cls, attention

@dataclass
class EnhancedEmbedding:
    """Enhanced embedding with transformer features"""
    text: str
    embedding: np.ndarray
    attention_weights: Optional[np.ndarray] = None
    token_embeddings: Optional[np.ndarray] = None
    model_used: str = "unknown"
    processing_time: float = 0.0
    confidence_score: float = 0.0

class EnhancedTransformerEngine:
    """Advanced transformer integration engine"""
    
    def __init__(self, config: TransformerConfig = None):
        """Initialize the enhanced transformer engine"""
        self.config = config or TransformerConfig()
        self.models = {}
        self.tokenizers = {}
        self.device = self._setup_device()
        self.embedding_cache = {}
        
        # Performance tracking
        self.processing_stats = {
            'total_texts_processed': 0,
            'total_processing_time': 0.0,
            'average_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"✅ Enhanced Transformer Engine initialized with device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup processing device (CPU/GPU)"""
        if self.config.device == "auto":
            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return self.config.device
    
    def _initialize_models(self):
        """Initialize transformer models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("⚠️ Transformers not available, using fallback models")
            self._initialize_fallback_models()
            return
        
        try:
            # Primary model initialization
            model_name = self.config.model_name
            logger.info(f"🔄 Loading transformer model: {model_name}")
            
            # Load model and tokenizer
            self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            self.models[model_name] = AutoModel.from_pretrained(model_name)
            
            # Move to device
            if self.device == "cuda":
                self.models[model_name] = self.models[model_name].to(self.device)
            
            # Set to evaluation mode
            self.models[model_name].eval()
            
            logger.info(f"✅ Transformer model loaded successfully: {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load transformer model: {e}")
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize fallback sentence transformer models"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.error("❌ No transformer models available")
            return
        
        try:
            # Fallback to sentence transformers
            fallback_models = [
                'all-MiniLM-L6-v2',
                'all-mpnet-base-v2',
                'paraphrase-MiniLM-L6-v2'
            ]
            
            for model_name in fallback_models:
                try:
                    self.models[model_name] = SentenceTransformer(model_name)
                    logger.info(f"✅ Fallback model loaded: {model_name}")
                    break
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {model_name}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Failed to initialize fallback models: {e}")
    
    def encode_text(self, 
                   text: Union[str, List[str]], 
                   model_name: Optional[str] = None,
                   return_attention: bool = None) -> Union[EnhancedEmbedding, List[EnhancedEmbedding]]:
        """
        Encode text using transformer models with enhanced features
        
        Args:
            text: Single text or list of texts to encode
            model_name: Specific model to use (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            Enhanced embedding(s) with transformer features
        """
        start_time = time.time()
        
        # Handle single text vs list
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        # Select model
        model_name = model_name or self.config.model_name
        if model_name not in self.models:
            model_name = list(self.models.keys())[0]
        
        # Check cache
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, txt in enumerate(texts):
            cache_key = f"{model_name}:{hash(txt)}"
            if self.config.cache_embeddings and cache_key in self.embedding_cache:
                embeddings.append(self.embedding_cache[cache_key])
                self.processing_stats['cache_hits'] += 1
            else:
                embeddings.append(None)
                uncached_texts.append(txt)
                uncached_indices.append(i)
                self.processing_stats['cache_misses'] += 1
        
        # Process uncached texts
        if uncached_texts:
            if TRANSFORMERS_AVAILABLE and model_name in self.models and hasattr(self.models[model_name], 'encode'):
                # Use transformer model
                new_embeddings = self._encode_with_transformers(
                    uncached_texts, model_name, return_attention
                )
            else:
                # Use sentence transformer fallback
                new_embeddings = self._encode_with_sentence_transformers(
                    uncached_texts, model_name
                )
            
            # Update embeddings and cache
            for i, embedding in zip(uncached_indices, new_embeddings):
                embeddings[i] = embedding
                if self.config.cache_embeddings:
                    cache_key = f"{model_name}:{hash(texts[i])}"
                    self.embedding_cache[cache_key] = embedding
        
        # Update stats
        processing_time = time.time() - start_time
        self.processing_stats['total_texts_processed'] += len(texts)
        self.processing_stats['total_processing_time'] += processing_time
        self.processing_stats['average_processing_time'] = (
            self.processing_stats['total_processing_time'] / 
            self.processing_stats['total_texts_processed']
        )
        
        return embeddings[0] if is_single else embeddings
    
    def _encode_with_transformers(self, 
                                 texts: List[str], 
                                 model_name: str,
                                 return_attention: bool = None) -> List[EnhancedEmbedding]:
        """Encode texts using transformer models"""
        if return_attention is None:
            return_attention = self.config.use_attention_weights
        
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=return_attention)
            
            # Extract embeddings
            last_hidden_states = outputs.last_hidden_state
            attention_weights = outputs.attentions if return_attention else None
            
            # Apply pooling strategy
            for j, text in enumerate(batch_texts):
                if self.config.pooling_strategy == "mean":
                    # Mean pooling
                    mask = inputs['attention_mask'][j].unsqueeze(-1)
                    embedding = (last_hidden_states[j] * mask).sum(0) / mask.sum(0)
                elif self.config.pooling_strategy == "max":
                    # Max pooling
                    embedding = torch.max(last_hidden_states[j], dim=0)[0]
                elif self.config.pooling_strategy == "cls":
                    # CLS token
                    embedding = last_hidden_states[j][0]
                else:
                    # Default to mean
                    mask = inputs['attention_mask'][j].unsqueeze(-1)
                    embedding = (last_hidden_states[j] * mask).sum(0) / mask.sum(0)
                
                # Convert to numpy
                embedding_np = embedding.cpu().numpy()
                
                # Extract attention weights if requested
                attention_np = None
                if attention_weights:
                    # Average attention across heads and layers
                    attention_tensor = torch.stack([layer[j] for layer in attention_weights])
                    attention_np = attention_tensor.mean(dim=(0, 1)).cpu().numpy()
                
                # Calculate confidence score (based on attention entropy)
                confidence = self._calculate_confidence(attention_np) if attention_np is not None else 0.8
                
                embeddings.append(EnhancedEmbedding(
                    text=text,
                    embedding=embedding_np,
                    attention_weights=attention_np,
                    model_used=model_name,
                    confidence_score=confidence
                ))
        
        return embeddings
    
    def _encode_with_sentence_transformers(self, 
                                         texts: List[str], 
                                         model_name: str) -> List[EnhancedEmbedding]:
        """Encode texts using sentence transformer models"""
        model = self.models[model_name]
        
        # Encode texts
        embeddings_np = model.encode(texts, batch_size=self.config.batch_size)
        
        # Create enhanced embeddings
        embeddings = []
        for text, embedding in zip(texts, embeddings_np):
            embeddings.append(EnhancedEmbedding(
                text=text,
                embedding=embedding,
                model_used=model_name,
                confidence_score=0.8  # Default confidence for sentence transformers
            ))
        
        return embeddings
    
    def _calculate_confidence(self, attention_weights: np.ndarray) -> float:
        """Calculate confidence score based on attention patterns"""
        if attention_weights is None:
            return 0.8
        
        # Calculate attention entropy (lower entropy = higher confidence)
        attention_probs = attention_weights / attention_weights.sum()
        entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-10))
        
        # Normalize to 0-1 range (lower entropy = higher confidence)
        max_entropy = np.log(len(attention_probs))
        confidence = 1.0 - (entropy / max_entropy)
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def compute_semantic_similarity(self, 
                                  text1: str, 
                                  text2: str,
                                  model_name: Optional[str] = None) -> float:
        """Compute semantic similarity between two texts"""
        # Get embeddings
        emb1 = self.encode_text(text1, model_name)
        emb2 = self.encode_text(text2, model_name)
        
        # Compute cosine similarity
        similarity = cosine_similarity(
            emb1.embedding.reshape(1, -1),
            emb2.embedding.reshape(1, -1)
        )[0, 0]
        
        return float(similarity)
    
    def cluster_texts(self, 
                     texts: List[str], 
                     n_clusters: int = 5,
                     model_name: Optional[str] = None) -> Dict[str, Any]:
        """Cluster texts using transformer embeddings"""
        # Get embeddings
        embeddings = self.encode_text(texts, model_name)
        embedding_matrix = np.vstack([emb.embedding for emb in embeddings])
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embedding_matrix)
        
        # Organize results
        clusters = {}
        for i, (text, label) in enumerate(zip(texts, cluster_labels)):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append({
                'text': text,
                'embedding': embeddings[i],
                'distance_to_center': float(
                    cosine_similarity(
                        embeddings[i].embedding.reshape(1, -1),
                        kmeans.cluster_centers_[label].reshape(1, -1)
                    )[0, 0]
                )
            })
        
        return {
            'clusters': clusters,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'n_clusters': n_clusters
        }
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        cache_hit_rate = (
            self.processing_stats['cache_hits'] / 
            (self.processing_stats['cache_hits'] + self.processing_stats['cache_misses'])
            if (self.processing_stats['cache_hits'] + self.processing_stats['cache_misses']) > 0
            else 0.0
        )
        
        return {
            **self.processing_stats,
            'cache_hit_rate': cache_hit_rate,
            'models_loaded': list(self.models.keys()),
            'device': self.device,
            'cache_size': len(self.embedding_cache)
        }
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache.clear()
        logger.info("🗑️ Embedding cache cleared")
    
    def save_cache(self, filepath: str):
        """Save embedding cache to file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            logger.info(f"💾 Cache saved to {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to save cache: {e}")
    
    def load_cache(self, filepath: str):
        """Load embedding cache from file"""
        try:
            with open(filepath, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            logger.info(f"📂 Cache loaded from {filepath}")
        except Exception as e:
            logger.error(f"❌ Failed to load cache: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Test the enhanced transformer engine
    config = TransformerConfig(
        model_name="distilbert-base-uncased",
        max_length=256,
        batch_size=8
    )
    
    engine = EnhancedTransformerEngine(config)
    
    # Test encoding
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy dog.",
        "Machine learning is transforming natural language processing.",
        "AI models are revolutionizing text understanding."
    ]
    
    print("🧪 Testing Enhanced Transformer Engine...")
    
    # Test single text encoding
    embedding = engine.encode_text(test_texts[0])
    print(f"✅ Single text encoded: {embedding.embedding.shape}")
    
    # Test batch encoding
    embeddings = engine.encode_text(test_texts)
    print(f"✅ Batch encoded: {len(embeddings)} texts")
    
    # Test similarity
    similarity = engine.compute_semantic_similarity(test_texts[0], test_texts[1])
    print(f"✅ Similarity computed: {similarity:.3f}")
    
    # Test clustering
    clusters = engine.cluster_texts(test_texts, n_clusters=2)
    print(f"✅ Clustering completed: {len(clusters['clusters'])} clusters")
    
    # Print stats
    stats = engine.get_processing_stats()
    print(f"📊 Processing stats: {stats}")
    
    print("🎉 Enhanced Transformer Engine test completed!")

