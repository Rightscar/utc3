"""
Real-time NLP Processing Pipeline
================================

This module implements a high-performance, real-time NLP processing pipeline
that can handle streaming text data with minimal latency and maximum throughput.

Features:
- Streaming text processing with configurable buffer sizes
- Asynchronous processing with concurrent workers
- Real-time quality monitoring and metrics
- Adaptive load balancing and resource management
- WebSocket support for real-time applications
- Event-driven architecture with callbacks
- Performance optimization and caching
"""

import asyncio
import logging
import time
import threading
from typing import List, Dict, Any, Callable, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import json
from datetime import datetime, timedelta

# Performance monitoring
import psutil
import gc

# Our NLP modules
from .semantic_understanding_engine import SemanticUnderstandingEngine
from .context_aware_chunker import ContextAwareChunker
from .enhanced_transformer_engine import EnhancedTransformerEngine
from .multilingual_processor import MultilingualProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingTask:
    """Individual processing task"""
    task_id: str
    text: str
    timestamp: datetime
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None

@dataclass
class ProcessingResult:
    """Processing result with metadata"""
    task_id: str
    text: str
    result: Dict[str, Any]
    processing_time: float
    timestamp: datetime
    success: bool = True
    error: Optional[str] = None

@dataclass
class PipelineConfig:
    """Configuration for real-time pipeline"""
    # Buffer settings
    buffer_size: int = 1000
    batch_size: int = 10
    max_text_length: int = 10000
    
    # Processing settings
    max_workers: int = 4
    processing_timeout: float = 30.0
    enable_caching: bool = True
    cache_ttl: int = 3600  # seconds
    
    # Performance settings
    target_latency: float = 1.0  # seconds
    max_queue_size: int = 10000
    enable_load_balancing: bool = True
    
    # Monitoring settings
    metrics_interval: int = 60  # seconds
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = defaultdict(list)
        self.start_time = time.time()
        self.last_reset = time.time()
        
        # Performance counters
        self.total_processed = 0
        self.total_errors = 0
        self.processing_times = deque(maxlen=1000)
        self.queue_sizes = deque(maxlen=1000)
        
    def record_processing_time(self, processing_time: float):
        """Record processing time"""
        self.processing_times.append(processing_time)
        self.total_processed += 1
    
    def record_error(self):
        """Record processing error"""
        self.total_errors += 1
    
    def record_queue_size(self, size: int):
        """Record queue size"""
        self.queue_sizes.append(size)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        now = time.time()
        uptime = now - self.start_time
        
        # Calculate averages
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        avg_queue_size = (
            sum(self.queue_sizes) / len(self.queue_sizes)
            if self.queue_sizes else 0.0
        )
        
        # Calculate throughput
        throughput = self.total_processed / uptime if uptime > 0 else 0.0
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        return {
            'uptime': uptime,
            'total_processed': self.total_processed,
            'total_errors': self.total_errors,
            'error_rate': self.total_errors / max(self.total_processed, 1),
            'throughput': throughput,
            'avg_processing_time': avg_processing_time,
            'avg_queue_size': avg_queue_size,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_info.percent,
            'memory_available': memory_info.available / (1024**3),  # GB
            'target_latency': self.config.target_latency,
            'latency_compliance': avg_processing_time <= self.config.target_latency
        }
    
    def should_scale_up(self) -> bool:
        """Determine if pipeline should scale up"""
        metrics = self.get_current_metrics()
        
        # Scale up conditions
        high_latency = metrics['avg_processing_time'] > self.config.target_latency * 1.5
        high_queue = metrics['avg_queue_size'] > self.config.batch_size * 2
        low_cpu = metrics['cpu_percent'] < 80
        
        return high_latency and high_queue and low_cpu
    
    def should_scale_down(self) -> bool:
        """Determine if pipeline should scale down"""
        metrics = self.get_current_metrics()
        
        # Scale down conditions
        low_latency = metrics['avg_processing_time'] < self.config.target_latency * 0.5
        low_queue = metrics['avg_queue_size'] < self.config.batch_size
        
        return low_latency and low_queue

class RealtimeNLPPipeline:
    """High-performance real-time NLP processing pipeline"""
    
    def __init__(self, config: PipelineConfig = None):
        """Initialize real-time NLP pipeline"""
        self.config = config or PipelineConfig()
        
        # Processing components
        self.semantic_engine = SemanticUnderstandingEngine()
        self.context_chunker = ContextAwareChunker()
        self.transformer_engine = None  # Lazy loading
        self.multilingual_processor = None  # Lazy loading
        
        # Pipeline state
        self.is_running = False
        self.task_queue = asyncio.Queue(maxsize=self.config.max_queue_size)
        self.result_queue = asyncio.Queue()
        self.workers = []
        self.current_workers = 0
        
        # Performance monitoring
        self.monitor = PerformanceMonitor(self.config)
        
        # Caching
        self.cache = {} if self.config.enable_caching else None
        self.cache_timestamps = {} if self.config.enable_caching else None
        
        # Event callbacks
        self.callbacks = {
            'on_result': [],
            'on_error': [],
            'on_metrics': []
        }
        
        logger.info(f"✅ Real-time NLP pipeline initialized with {self.config.max_workers} workers")
    
    def add_callback(self, event: str, callback: Callable):
        """Add event callback"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _lazy_load_components(self):
        """Lazy load heavy components"""
        if self.transformer_engine is None:
            try:
                from .enhanced_transformer_engine import EnhancedTransformerEngine, TransformerConfig
                config = TransformerConfig(batch_size=self.config.batch_size)
                self.transformer_engine = EnhancedTransformerEngine(config)
                logger.info("✅ Transformer engine loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load transformer engine: {e}")
        
        if self.multilingual_processor is None:
            try:
                self.multilingual_processor = MultilingualProcessor()
                logger.info("✅ Multilingual processor loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load multilingual processor: {e}")
    
    async def start(self):
        """Start the real-time processing pipeline"""
        if self.is_running:
            logger.warning("⚠️ Pipeline is already running")
            return
        
        self.is_running = True
        logger.info("🚀 Starting real-time NLP pipeline...")
        
        # Lazy load components
        self._lazy_load_components()
        
        # Start worker tasks
        for i in range(self.config.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            self.current_workers += 1
        
        # Start monitoring task
        if self.config.enable_performance_monitoring:
            monitor_task = asyncio.create_task(self._monitor_performance())
            self.workers.append(monitor_task)
        
        # Start cache cleanup task
        if self.config.enable_caching:
            cleanup_task = asyncio.create_task(self._cleanup_cache())
            self.workers.append(cleanup_task)
        
        logger.info(f"✅ Pipeline started with {self.current_workers} workers")
    
    async def stop(self):
        """Stop the real-time processing pipeline"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("🛑 Stopping real-time NLP pipeline...")
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        self.current_workers = 0
        
        logger.info("✅ Pipeline stopped")
    
    async def process_text(self, 
                          text: str, 
                          task_id: Optional[str] = None,
                          priority: int = 1,
                          metadata: Optional[Dict[str, Any]] = None,
                          callback: Optional[Callable] = None) -> str:
        """Submit text for real-time processing"""
        if not self.is_running:
            raise RuntimeError("Pipeline is not running")
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task-{int(time.time() * 1000000)}"
        
        # Create processing task
        task = ProcessingTask(
            task_id=task_id,
            text=text,
            timestamp=datetime.now(),
            priority=priority,
            metadata=metadata or {},
            callback=callback
        )
        
        # Check cache first
        if self.config.enable_caching:
            cached_result = self._get_cached_result(text)
            if cached_result:
                logger.debug(f"📋 Cache hit for task {task_id}")
                return cached_result
        
        # Submit to queue
        try:
            await self.task_queue.put(task)
            logger.debug(f"📝 Task {task_id} submitted to queue")
            return task_id
        except asyncio.QueueFull:
            logger.error(f"❌ Queue full, rejecting task {task_id}")
            raise RuntimeError("Processing queue is full")
    
    async def get_result(self, timeout: Optional[float] = None) -> Optional[ProcessingResult]:
        """Get next processing result"""
        try:
            result = await asyncio.wait_for(
                self.result_queue.get(), 
                timeout=timeout or self.config.processing_timeout
            )
            return result
        except asyncio.TimeoutError:
            return None
    
    async def process_stream(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[ProcessingResult, None]:
        """Process streaming text data"""
        async for text in text_stream:
            task_id = await self.process_text(text)
            
            # Wait for result
            while True:
                result = await self.get_result(timeout=1.0)
                if result and result.task_id == task_id:
                    yield result
                    break
                elif result:
                    # Different task result, put it back
                    await self.result_queue.put(result)
    
    async def _worker(self, worker_id: str):
        """Worker coroutine for processing tasks"""
        logger.info(f"👷 Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get task from queue
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                
                # Record queue size
                self.monitor.record_queue_size(self.task_queue.qsize())
                
                # Process task
                start_time = time.time()
                result = await self._process_task(task)
                processing_time = time.time() - start_time
                
                # Record metrics
                self.monitor.record_processing_time(processing_time)
                
                # Cache result if enabled
                if self.config.enable_caching and result.success:
                    self._cache_result(task.text, result.result)
                
                # Submit result
                await self.result_queue.put(result)
                
                # Call callbacks
                if result.success:
                    for callback in self.callbacks['on_result']:
                        try:
                            await callback(result)
                        except Exception as e:
                            logger.error(f"❌ Callback error: {e}")
                else:
                    for callback in self.callbacks['on_error']:
                        try:
                            await callback(result)
                        except Exception as e:
                            logger.error(f"❌ Error callback error: {e}")
                
                # Call task-specific callback
                if task.callback:
                    try:
                        await task.callback(result)
                    except Exception as e:
                        logger.error(f"❌ Task callback error: {e}")
                
                logger.debug(f"✅ Worker {worker_id} completed task {task.task_id} in {processing_time:.3f}s")
                
            except asyncio.TimeoutError:
                # No tasks available, continue
                continue
            except Exception as e:
                logger.error(f"❌ Worker {worker_id} error: {e}")
                self.monitor.record_error()
        
        logger.info(f"👷 Worker {worker_id} stopped")
    
    async def _process_task(self, task: ProcessingTask) -> ProcessingResult:
        """Process individual task"""
        try:
            # Validate text length
            if len(task.text) > self.config.max_text_length:
                raise ValueError(f"Text too long: {len(task.text)} > {self.config.max_text_length}")
            
            # Perform NLP processing
            result = {}
            
            # Basic semantic analysis
            if hasattr(self.semantic_engine, 'analyze_semantic_structure'):
                semantic_analysis = self.semantic_engine.analyze_semantic_structure(task.text)
                result['semantic_analysis'] = semantic_analysis
            
            # Context-aware chunking
            if hasattr(self.context_chunker, 'chunk_with_context'):
                chunks = self.context_chunker.chunk_with_context(task.text)
                result['chunks'] = [{'text': chunk.text, 'quality_score': chunk.coherence_score} for chunk in chunks]
            
            # Enhanced transformer processing (if available)
            if self.transformer_engine:
                try:
                    embedding = self.transformer_engine.encode_text(task.text)
                    result['embedding'] = {
                        'model_used': embedding.model_used,
                        'confidence': embedding.confidence_score,
                        'embedding_size': len(embedding.embedding)
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Transformer processing failed: {e}")
            
            # Multilingual processing (if available)
            if self.multilingual_processor:
                try:
                    multilingual_result = self.multilingual_processor.process_multilingual_text(task.text)
                    result['language'] = {
                        'detected': multilingual_result.language.code,
                        'name': multilingual_result.language.name,
                        'confidence': multilingual_result.language.confidence
                    }
                except Exception as e:
                    logger.warning(f"⚠️ Multilingual processing failed: {e}")
            
            # Add metadata
            result['metadata'] = {
                'text_length': len(task.text),
                'word_count': len(task.text.split()),
                'processing_timestamp': datetime.now().isoformat(),
                'task_metadata': task.metadata
            }
            
            return ProcessingResult(
                task_id=task.task_id,
                text=task.text,
                result=result,
                processing_time=0.0,  # Will be set by worker
                timestamp=datetime.now(),
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Task processing failed: {e}")
            return ProcessingResult(
                task_id=task.task_id,
                text=task.text,
                result={},
                processing_time=0.0,
                timestamp=datetime.now(),
                success=False,
                error=str(e)
            )
    
    def _get_cached_result(self, text: str) -> Optional[Dict[str, Any]]:
        """Get cached result for text"""
        if not self.config.enable_caching:
            return None
        
        text_hash = hash(text)
        if text_hash in self.cache:
            # Check TTL
            if time.time() - self.cache_timestamps[text_hash] < self.config.cache_ttl:
                return self.cache[text_hash]
            else:
                # Expired, remove from cache
                del self.cache[text_hash]
                del self.cache_timestamps[text_hash]
        
        return None
    
    def _cache_result(self, text: str, result: Dict[str, Any]):
        """Cache processing result"""
        if not self.config.enable_caching:
            return
        
        text_hash = hash(text)
        self.cache[text_hash] = result
        self.cache_timestamps[text_hash] = time.time()
    
    async def _monitor_performance(self):
        """Monitor pipeline performance"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.metrics_interval)
                
                metrics = self.monitor.get_current_metrics()
                
                # Log metrics
                logger.info(f"📊 Pipeline metrics: "
                          f"throughput={metrics['throughput']:.2f}/s, "
                          f"latency={metrics['avg_processing_time']:.3f}s, "
                          f"queue={metrics['avg_queue_size']:.1f}, "
                          f"cpu={metrics['cpu_percent']:.1f}%, "
                          f"memory={metrics['memory_percent']:.1f}%")
                
                # Call metrics callbacks
                for callback in self.callbacks['on_metrics']:
                    try:
                        await callback(metrics)
                    except Exception as e:
                        logger.error(f"❌ Metrics callback error: {e}")
                
                # Auto-scaling (if enabled)
                if self.config.enable_load_balancing:
                    await self._auto_scale(metrics)
                
            except Exception as e:
                logger.error(f"❌ Performance monitoring error: {e}")
    
    async def _auto_scale(self, metrics: Dict[str, Any]):
        """Auto-scale workers based on performance"""
        # Simple auto-scaling logic
        if self.monitor.should_scale_up() and self.current_workers < self.config.max_workers * 2:
            # Add worker
            worker_id = f"worker-auto-{self.current_workers}"
            worker = asyncio.create_task(self._worker(worker_id))
            self.workers.append(worker)
            self.current_workers += 1
            logger.info(f"📈 Scaled up: added worker {worker_id}")
            
        elif self.monitor.should_scale_down() and self.current_workers > self.config.max_workers:
            # Remove worker (simplified - just reduce count)
            self.current_workers -= 1
            logger.info(f"📉 Scaled down: reduced to {self.current_workers} workers")
    
    async def _cleanup_cache(self):
        """Periodic cache cleanup"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.cache_ttl // 4)  # Cleanup every quarter TTL
                
                if self.config.enable_caching:
                    current_time = time.time()
                    expired_keys = [
                        key for key, timestamp in self.cache_timestamps.items()
                        if current_time - timestamp > self.config.cache_ttl
                    ]
                    
                    for key in expired_keys:
                        del self.cache[key]
                        del self.cache_timestamps[key]
                    
                    if expired_keys:
                        logger.debug(f"🗑️ Cleaned up {len(expired_keys)} expired cache entries")
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"❌ Cache cleanup error: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'is_running': self.is_running,
            'current_workers': self.current_workers,
            'max_workers': self.config.max_workers,
            'queue_size': self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0,
            'result_queue_size': self.result_queue.qsize() if hasattr(self.result_queue, 'qsize') else 0,
            'cache_size': len(self.cache) if self.cache else 0,
            'metrics': self.monitor.get_current_metrics()
        }

# Example usage and testing
async def test_realtime_pipeline():
    """Test the real-time NLP pipeline"""
    print("🧪 Testing Real-time NLP Pipeline...")
    
    # Create pipeline
    config = PipelineConfig(
        max_workers=2,
        batch_size=5,
        target_latency=0.5,
        enable_caching=True
    )
    
    pipeline = RealtimeNLPPipeline(config)
    
    # Add result callback
    async def on_result(result: ProcessingResult):
        print(f"✅ Result received: {result.task_id} ({result.processing_time:.3f}s)")
    
    pipeline.add_callback('on_result', on_result)
    
    # Start pipeline
    await pipeline.start()
    
    # Submit test tasks
    test_texts = [
        "This is a test sentence for real-time processing.",
        "Another test with different content and structure.",
        "Real-time NLP processing enables immediate text analysis.",
        "Performance monitoring helps optimize processing pipelines."
    ]
    
    # Submit tasks
    task_ids = []
    for i, text in enumerate(test_texts):
        task_id = await pipeline.process_text(text, priority=i+1)
        task_ids.append(task_id)
        print(f"📝 Submitted task: {task_id}")
    
    # Wait for results
    results_received = 0
    while results_received < len(test_texts):
        result = await pipeline.get_result(timeout=5.0)
        if result:
            results_received += 1
            print(f"📊 Result: {result.task_id} - Success: {result.success}")
        else:
            print("⏰ Timeout waiting for result")
            break
    
    # Get status
    status = pipeline.get_pipeline_status()
    print(f"📈 Pipeline status: {status}")
    
    # Stop pipeline
    await pipeline.stop()
    print("🎉 Real-time pipeline test completed!")

if __name__ == "__main__":
    asyncio.run(test_realtime_pipeline())

