"""
Comprehensive Error Handling Module
==================================

This module provides robust error handling, input validation, connection management,
and graceful degradation for the Universal Text-to-Dialogue AI system.

Features:
- Input validation and sanitization
- Connection timeout management
- Model loading error handling
- Memory management and monitoring
- Output format validation
- User-friendly error messages
- Logging and debugging support
"""

import logging
import re
import time
import asyncio
import gc
import psutil
import traceback
from typing import Dict, Any, Tuple, Optional, Union, List
from dataclasses import dataclass
from functools import wraps
from contextlib import contextmanager
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ErrorInfo:
    """Error information container"""
    error_type: str
    message: str
    details: str
    timestamp: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    user_message: str
    recovery_suggestion: str

class InputValidator:
    """Comprehensive input validation and sanitization"""
    
    # Input limits and constraints
    MAX_TEXT_LENGTH = 1000000  # 1MB
    MIN_TEXT_LENGTH = 10
    MAX_CHUNK_SIZE = 10000
    MIN_CHUNK_SIZE = 50
    
    # Allowed characters pattern
    SAFE_TEXT_PATTERN = re.compile(r'^[\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\n\r\t]+$', re.UNICODE)
    
    @staticmethod
    def validate_text_input(text: str) -> Tuple[bool, str, str]:
        """
        Validate and sanitize text input
        
        Returns:
            (is_valid, sanitized_text, error_message)
        """
        try:
            # Check if text exists
            if not text:
                return False, "", "Text input cannot be empty"
            
            # Strip whitespace
            text = text.strip()
            
            if not text:
                return False, "", "Text input cannot be empty after removing whitespace"
            
            # Check length constraints
            if len(text) < InputValidator.MIN_TEXT_LENGTH:
                return False, text, f"Text too short (minimum {InputValidator.MIN_TEXT_LENGTH} characters)"
            
            if len(text) > InputValidator.MAX_TEXT_LENGTH:
                return False, text[:InputValidator.MAX_TEXT_LENGTH], f"Text too long (maximum {InputValidator.MAX_TEXT_LENGTH} characters), truncated"
            
            # Basic sanitization - remove potentially harmful content
            # Keep most Unicode characters but remove control characters
            sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'<script[^>]*>.*?</script>',  # Script tags
                r'javascript:',  # JavaScript URLs
                r'data:text/html',  # Data URLs
                r'vbscript:',  # VBScript
            ]
            
            for pattern in suspicious_patterns:
                if re.search(pattern, sanitized, re.IGNORECASE):
                    logger.warning(f"Suspicious content detected and removed: {pattern}")
                    sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
            
            return True, sanitized, ""
            
        except Exception as e:
            logger.error(f"Input validation error: {e}")
            return False, "", f"Input validation failed: {str(e)}"
    
    @staticmethod
    def validate_chunk_size(chunk_size: int) -> Tuple[bool, int, str]:
        """Validate chunk size parameter"""
        try:
            if not isinstance(chunk_size, int):
                chunk_size = int(chunk_size)
            
            if chunk_size < InputValidator.MIN_CHUNK_SIZE:
                return False, InputValidator.MIN_CHUNK_SIZE, f"Chunk size too small, using minimum: {InputValidator.MIN_CHUNK_SIZE}"
            
            if chunk_size > InputValidator.MAX_CHUNK_SIZE:
                return False, InputValidator.MAX_CHUNK_SIZE, f"Chunk size too large, using maximum: {InputValidator.MAX_CHUNK_SIZE}"
            
            return True, chunk_size, ""
            
        except (ValueError, TypeError) as e:
            logger.error(f"Chunk size validation error: {e}")
            return False, 500, f"Invalid chunk size, using default: 500"
    
    @staticmethod
    def validate_model_name(model_name: str) -> Tuple[bool, str, str]:
        """Validate transformer model name"""
        try:
            if not model_name or not model_name.strip():
                return False, "sentence-transformers/all-MiniLM-L6-v2", "Empty model name, using default"
            
            # List of known safe models
            safe_models = [
                "sentence-transformers/all-MiniLM-L6-v2",
                "sentence-transformers/all-mpnet-base-v2",
                "sentence-transformers/paraphrase-MiniLM-L6-v2",
                "bert-base-uncased",
                "distilbert-base-uncased",
                "roberta-base"
            ]
            
            model_name = model_name.strip()
            
            # Check if it's a known safe model
            if model_name in safe_models:
                return True, model_name, ""
            
            # Check for suspicious patterns in model names
            if any(char in model_name for char in ['<', '>', '&', '"', "'"]):
                return False, "sentence-transformers/all-MiniLM-L6-v2", "Suspicious model name, using default"
            
            # Allow the model name but log it
            logger.info(f"Using custom model: {model_name}")
            return True, model_name, f"Using custom model: {model_name}"
            
        except Exception as e:
            logger.error(f"Model name validation error: {e}")
            return False, "sentence-transformers/all-MiniLM-L6-v2", f"Model validation failed, using default: {str(e)}"

class ConnectionManager:
    """Manage connections and timeouts"""
    
    DEFAULT_TIMEOUT = 30
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    
    @staticmethod
    def with_timeout(timeout_seconds: int = DEFAULT_TIMEOUT):
        """Decorator for adding timeout to functions"""
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    error_msg = f"Operation timed out after {timeout_seconds} seconds"
                    logger.error(error_msg)
                    raise TimeoutError(error_msg)
                except Exception as e:
                    logger.error(f"Operation failed: {e}")
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    elapsed = time.time() - start_time
                    if elapsed > timeout_seconds:
                        error_msg = f"Operation exceeded timeout ({elapsed:.2f}s > {timeout_seconds}s)"
                        logger.warning(error_msg)
                    return result
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.error(f"Operation failed after {elapsed:.2f}s: {e}")
                    raise
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    @staticmethod
    def with_retry(max_retries: int = MAX_RETRIES, delay: float = RETRY_DELAY):
        """Decorator for adding retry logic"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        if attempt < max_retries:
                            logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay}s...")
                            time.sleep(delay)
                        else:
                            logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
                
                raise last_exception
            
            return wrapper
        return decorator

class ModelManager:
    """Manage model loading with error handling and fallbacks"""
    
    @staticmethod
    @ConnectionManager.with_timeout(60)  # 60 second timeout for model loading
    @ConnectionManager.with_retry(max_retries=2)
    def load_model_with_fallback(model_name: str) -> Tuple[Any, str, bool]:
        """
        Load model with graceful fallback
        
        Returns:
            (model, status_message, is_fallback)
        """
        try:
            # Validate model name first
            is_valid, validated_name, validation_msg = InputValidator.validate_model_name(model_name)
            
            if not is_valid:
                logger.warning(f"Model validation failed: {validation_msg}")
                model_name = validated_name
            
            # Try to load the requested model
            logger.info(f"Loading model: {model_name}")
            
            # Import here to avoid circular imports
            try:
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name)
                return model, f"✅ {model_name} loaded successfully", False
                
            except Exception as model_error:
                logger.warning(f"Failed to load {model_name}: {model_error}")
                
                # Try fallback model
                fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
                if model_name != fallback_model:
                    logger.info(f"Attempting fallback to: {fallback_model}")
                    fallback = SentenceTransformer(fallback_model)
                    return fallback, f"⚠️ Using fallback model due to: {str(model_error)}", True
                else:
                    raise model_error
                    
        except Exception as e:
            error_msg = f"Failed to load any model: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

class MemoryManager:
    """Monitor and manage memory usage"""
    
    MEMORY_WARNING_THRESHOLD = 85  # Percentage
    MEMORY_CRITICAL_THRESHOLD = 95  # Percentage
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get current memory usage information"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            }
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {'total_gb': 0, 'available_gb': 0, 'used_gb': 0, 'percent': 0}
    
    @staticmethod
    def check_memory_usage() -> Tuple[bool, str]:
        """Check memory usage and return warning if needed"""
        try:
            memory_info = MemoryManager.get_memory_info()
            percent = memory_info['percent']
            
            if percent > MemoryManager.MEMORY_CRITICAL_THRESHOLD:
                gc.collect()  # Force garbage collection
                return False, f"🔴 Critical memory usage: {percent:.1f}% - Forced cleanup"
            
            elif percent > MemoryManager.MEMORY_WARNING_THRESHOLD:
                return True, f"🟡 High memory usage: {percent:.1f}% - Consider reducing batch size"
            
            else:
                return True, f"🟢 Memory usage normal: {percent:.1f}%"
                
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return True, "⚠️ Unable to check memory usage"
    
    @staticmethod
    @contextmanager
    def memory_monitor(operation_name: str):
        """Context manager for monitoring memory during operations"""
        start_memory = MemoryManager.get_memory_info()
        start_time = time.time()
        
        try:
            logger.info(f"Starting {operation_name} - Memory: {start_memory['percent']:.1f}%")
            yield
        finally:
            end_memory = MemoryManager.get_memory_info()
            end_time = time.time()
            
            memory_delta = end_memory['percent'] - start_memory['percent']
            time_delta = end_time - start_time
            
            logger.info(f"Completed {operation_name} - "
                       f"Time: {time_delta:.2f}s, "
                       f"Memory change: {memory_delta:+.1f}%, "
                       f"Final: {end_memory['percent']:.1f}%")

class OutputValidator:
    """Validate output formats and data structures"""
    
    @staticmethod
    def validate_processing_result(result: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate processing result format"""
        try:
            # Required fields for processing results
            required_fields = ['chunks', 'metadata', 'quality_scores']
            
            for field in required_fields:
                if field not in result:
                    return False, f"Missing required field: {field}"
            
            # Validate chunks structure
            chunks = result.get('chunks', [])
            if not isinstance(chunks, list):
                return False, "Chunks must be a list"
            
            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, dict):
                    return False, f"Chunk {i} must be a dictionary"
                
                if 'text' not in chunk:
                    return False, f"Chunk {i} missing 'text' field"
                
                if 'quality_score' not in chunk:
                    return False, f"Chunk {i} missing 'quality_score' field"
            
            # Validate metadata structure
            metadata = result.get('metadata', {})
            if not isinstance(metadata, dict):
                return False, "Metadata must be a dictionary"
            
            # Validate quality scores
            quality_scores = result.get('quality_scores', {})
            if not isinstance(quality_scores, dict):
                return False, "Quality scores must be a dictionary"
            
            return True, "Output validation passed"
            
        except Exception as e:
            logger.error(f"Output validation error: {e}")
            return False, f"Output validation failed: {str(e)}"
    
    @staticmethod
    def sanitize_output(result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize output to ensure safe data"""
        try:
            sanitized = {}
            
            # Sanitize chunks
            if 'chunks' in result:
                sanitized_chunks = []
                for chunk in result['chunks']:
                    if isinstance(chunk, dict) and 'text' in chunk:
                        sanitized_chunk = {
                            'text': str(chunk['text'])[:10000],  # Limit text length
                            'quality_score': float(chunk.get('quality_score', 0.0))
                        }
                        sanitized_chunks.append(sanitized_chunk)
                sanitized['chunks'] = sanitized_chunks
            
            # Sanitize metadata
            if 'metadata' in result:
                metadata = result['metadata']
                if isinstance(metadata, dict):
                    sanitized['metadata'] = {
                        k: str(v)[:1000] if isinstance(v, str) else v
                        for k, v in metadata.items()
                    }
            
            # Sanitize quality scores
            if 'quality_scores' in result:
                quality_scores = result['quality_scores']
                if isinstance(quality_scores, dict):
                    sanitized['quality_scores'] = {
                        k: float(v) if isinstance(v, (int, float)) else 0.0
                        for k, v in quality_scores.items()
                    }
            
            return sanitized
            
        except Exception as e:
            logger.error(f"Output sanitization error: {e}")
            return {'chunks': [], 'metadata': {}, 'quality_scores': {}}

class ErrorHandler:
    """Main error handling coordinator"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.max_history = 100
    
    def handle_error(self, 
                    error: Exception, 
                    context: str = "",
                    severity: str = "medium",
                    user_friendly: bool = True) -> ErrorInfo:
        """Handle and log errors with user-friendly messages"""
        
        error_type = type(error).__name__
        error_message = str(error)
        error_details = traceback.format_exc()
        
        # Generate user-friendly message
        user_message = self._generate_user_message(error_type, error_message, context)
        recovery_suggestion = self._generate_recovery_suggestion(error_type, context)
        
        # Create error info
        error_info = ErrorInfo(
            error_type=error_type,
            message=error_message,
            details=error_details,
            timestamp=time.time(),
            severity=severity,
            user_message=user_message,
            recovery_suggestion=recovery_suggestion
        )
        
        # Log error
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity, logging.ERROR)
        
        logger.log(log_level, f"Error in {context}: {error_message}")
        
        # Store in history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        # Display to user if in Streamlit context
        if user_friendly:
            self._display_error_to_user(error_info)
        
        return error_info
    
    def _generate_user_message(self, error_type: str, error_message: str, context: str) -> str:
        """Generate user-friendly error messages"""
        
        user_messages = {
            'TimeoutError': "⏰ The operation is taking longer than expected. Please try again with a smaller text or check your internet connection.",
            'MemoryError': "💾 Not enough memory to process this request. Please try with a smaller text or restart the application.",
            'ConnectionError': "🌐 Unable to connect to required services. Please check your internet connection and try again.",
            'FileNotFoundError': "📁 Required file not found. Please ensure all necessary files are available.",
            'ValueError': "⚠️ Invalid input provided. Please check your input and try again.",
            'RuntimeError': "🔧 A processing error occurred. Please try again or contact support if the problem persists.",
            'ImportError': "📦 Required component not available. Please ensure all dependencies are installed.",
            'KeyError': "🔑 Missing required information. Please check your input format.",
            'TypeError': "🔤 Incorrect data type provided. Please check your input format.",
        }
        
        base_message = user_messages.get(error_type, "❌ An unexpected error occurred. Please try again.")
        
        if context:
            return f"{base_message} (Context: {context})"
        
        return base_message
    
    def _generate_recovery_suggestion(self, error_type: str, context: str) -> str:
        """Generate recovery suggestions"""
        
        suggestions = {
            'TimeoutError': "Try reducing the text size or check your internet connection.",
            'MemoryError': "Reduce the text size, close other applications, or restart the app.",
            'ConnectionError': "Check your internet connection and try again.",
            'FileNotFoundError': "Ensure all required files are in the correct location.",
            'ValueError': "Check your input format and ensure all required fields are filled.",
            'RuntimeError': "Try again with different settings or restart the application.",
            'ImportError': "Reinstall the application or check system requirements.",
            'KeyError': "Ensure all required input fields are provided.",
            'TypeError': "Check that your input is in the correct format.",
        }
        
        return suggestions.get(error_type, "Try again or contact support if the problem persists.")
    
    def _display_error_to_user(self, error_info: ErrorInfo):
        """Display error to user in Streamlit interface"""
        try:
            if error_info.severity == 'critical':
                st.error(f"🚨 {error_info.user_message}")
                st.error(f"💡 **Suggestion**: {error_info.recovery_suggestion}")
            elif error_info.severity == 'high':
                st.error(f"❌ {error_info.user_message}")
                st.info(f"💡 **Suggestion**: {error_info.recovery_suggestion}")
            elif error_info.severity == 'medium':
                st.warning(f"⚠️ {error_info.user_message}")
                st.info(f"💡 **Suggestion**: {error_info.recovery_suggestion}")
            else:
                st.info(f"ℹ️ {error_info.user_message}")
        except:
            # Fallback if Streamlit is not available
            print(f"Error: {error_info.user_message}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors"""
        if not self.error_history:
            return {'total_errors': 0, 'recent_errors': [], 'error_types': {}}
        
        recent_errors = self.error_history[-10:]  # Last 10 errors
        error_types = {}
        
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        
        return {
            'total_errors': len(self.error_history),
            'recent_errors': [
                {
                    'type': error.error_type,
                    'message': error.user_message,
                    'timestamp': error.timestamp,
                    'severity': error.severity
                }
                for error in recent_errors
            ],
            'error_types': error_types
        }

# Global error handler instance
global_error_handler = ErrorHandler()

# Convenience functions
def handle_error(error: Exception, context: str = "", severity: str = "medium") -> ErrorInfo:
    """Convenience function for error handling"""
    return global_error_handler.handle_error(error, context, severity)

def validate_input(text: str) -> Tuple[bool, str, str]:
    """Convenience function for input validation"""
    return InputValidator.validate_text_input(text)

def check_memory() -> Tuple[bool, str]:
    """Convenience function for memory checking"""
    return MemoryManager.check_memory_usage()

def validate_output(result: Dict[str, Any]) -> Tuple[bool, str]:
    """Convenience function for output validation"""
    return OutputValidator.validate_processing_result(result)

# Example usage and testing
if __name__ == "__main__":
    print("🧪 Testing Error Handling Module...")
    
    # Test input validation
    valid, sanitized, error = validate_input("This is a test text for validation.")
    print(f"Input validation: {valid}, Error: {error}")
    
    # Test memory checking
    memory_ok, memory_msg = check_memory()
    print(f"Memory check: {memory_msg}")
    
    # Test error handling
    try:
        raise ValueError("Test error for demonstration")
    except Exception as e:
        error_info = handle_error(e, "testing", "low")
        print(f"Error handled: {error_info.user_message}")
    
    # Test output validation
    test_result = {
        'chunks': [{'text': 'test', 'quality_score': 0.8}],
        'metadata': {'test': 'value'},
        'quality_scores': {'overall': 0.8}
    }
    
    output_valid, output_error = validate_output(test_result)
    print(f"Output validation: {output_valid}, Error: {output_error}")
    
    print("🎉 Error handling module test completed!")

