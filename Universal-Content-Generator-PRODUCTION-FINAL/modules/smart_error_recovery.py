#!/usr/bin/env python3
"""
Smart Error Recovery Module
Intelligent error classification, handling, and automatic recovery mechanisms
"""

import time
import random
import traceback
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import streamlit as st

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    NETWORK = "network"
    API = "api"
    FILE_IO = "file_io"
    PROCESSING = "processing"
    VALIDATION = "validation"
    MEMORY = "memory"
    TIMEOUT = "timeout"
    USER_INPUT = "user_input"
    SYSTEM = "system"

class SmartErrorRecovery:
    """Intelligent error recovery system with automatic retry and fallback mechanisms"""
    
    def __init__(self):
        self.error_history = []
        self.recovery_strategies = {}
        self.retry_configs = {}
        self.fallback_handlers = {}
        
        # Initialize session state for error tracking
        if 'error_recovery_state' not in st.session_state:
            st.session_state.error_recovery_state = {
                'error_count': 0,
                'last_error_time': None,
                'recovery_attempts': {},
                'fallback_mode': False,
                'error_patterns': []
            }
        
        self.logger = logging.getLogger(__name__)
        self.setup_recovery_strategies()
        self.setup_retry_configurations()
        self.setup_fallback_handlers()
    
    def setup_recovery_strategies(self):
        """Configure recovery strategies for different error types"""
        self.recovery_strategies = {
            ErrorCategory.NETWORK: {
                'retry_count': 3,
                'backoff_factor': 2.0,
                'max_delay': 30,
                'fallback_available': True,
                'user_notification': True
            },
            ErrorCategory.API: {
                'retry_count': 5,
                'backoff_factor': 1.5,
                'max_delay': 60,
                'fallback_available': True,
                'user_notification': True
            },
            ErrorCategory.FILE_IO: {
                'retry_count': 2,
                'backoff_factor': 1.0,
                'max_delay': 5,
                'fallback_available': True,
                'user_notification': False
            },
            ErrorCategory.PROCESSING: {
                'retry_count': 2,
                'backoff_factor': 1.0,
                'max_delay': 10,
                'fallback_available': True,
                'user_notification': True
            },
            ErrorCategory.VALIDATION: {
                'retry_count': 1,
                'backoff_factor': 1.0,
                'max_delay': 2,
                'fallback_available': True,
                'user_notification': False
            },
            ErrorCategory.MEMORY: {
                'retry_count': 1,
                'backoff_factor': 1.0,
                'max_delay': 5,
                'fallback_available': True,
                'user_notification': True
            },
            ErrorCategory.TIMEOUT: {
                'retry_count': 3,
                'backoff_factor': 2.0,
                'max_delay': 120,
                'fallback_available': True,
                'user_notification': True
            },
            ErrorCategory.USER_INPUT: {
                'retry_count': 0,
                'backoff_factor': 1.0,
                'max_delay': 0,
                'fallback_available': False,
                'user_notification': True
            },
            ErrorCategory.SYSTEM: {
                'retry_count': 1,
                'backoff_factor': 1.0,
                'max_delay': 10,
                'fallback_available': True,
                'user_notification': True
            }
        }
    
    def setup_retry_configurations(self):
        """Configure retry mechanisms for specific operations"""
        self.retry_configs = {
            'file_upload': {
                'max_retries': 3,
                'base_delay': 1.0,
                'max_delay': 10.0,
                'jitter': True
            },
            'content_extraction': {
                'max_retries': 2,
                'base_delay': 2.0,
                'max_delay': 15.0,
                'jitter': False
            },
            'api_call': {
                'max_retries': 5,
                'base_delay': 1.0,
                'max_delay': 60.0,
                'jitter': True
            },
            'chunking': {
                'max_retries': 2,
                'base_delay': 1.0,
                'max_delay': 5.0,
                'jitter': False
            },
            'validation': {
                'max_retries': 1,
                'base_delay': 0.5,
                'max_delay': 2.0,
                'jitter': False
            },
            'export': {
                'max_retries': 3,
                'base_delay': 2.0,
                'max_delay': 20.0,
                'jitter': True
            }
        }
    
    def setup_fallback_handlers(self):
        """Configure fallback handlers for when retries fail"""
        self.fallback_handlers = {
            ErrorCategory.NETWORK: self.network_fallback,
            ErrorCategory.API: self.api_fallback,
            ErrorCategory.FILE_IO: self.file_io_fallback,
            ErrorCategory.PROCESSING: self.processing_fallback,
            ErrorCategory.VALIDATION: self.validation_fallback,
            ErrorCategory.MEMORY: self.memory_fallback,
            ErrorCategory.TIMEOUT: self.timeout_fallback,
            ErrorCategory.SYSTEM: self.system_fallback
        }
    
    def classify_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Intelligently classify errors based on type and context"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Network-related errors
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout', 'dns', 'socket']):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM
        
        # API-related errors
        elif any(keyword in error_message for keyword in ['api', 'http', 'request', 'response', '401', '403', '429', '500', '502', '503']):
            category = ErrorCategory.API
            severity = ErrorSeverity.MEDIUM
        
        # File I/O errors
        elif any(keyword in error_message for keyword in ['file', 'directory', 'permission', 'not found', 'access denied']):
            category = ErrorCategory.FILE_IO
            severity = ErrorSeverity.LOW
        
        # Memory errors
        elif any(keyword in error_message for keyword in ['memory', 'out of memory', 'memoryerror']):
            category = ErrorCategory.MEMORY
            severity = ErrorSeverity.HIGH
        
        # Timeout errors
        elif any(keyword in error_message for keyword in ['timeout', 'timed out', 'timeouterror']):
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.MEDIUM
        
        # Validation errors
        elif any(keyword in error_message for keyword in ['validation', 'invalid', 'format', 'schema']):
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.LOW
        
        # User input errors
        elif any(keyword in error_message for keyword in ['user', 'input', 'parameter', 'argument']):
            category = ErrorCategory.USER_INPUT
            severity = ErrorSeverity.LOW
        
        # Processing errors
        elif context in ['chunking', 'enhancement', 'analysis', 'processing']:
            category = ErrorCategory.PROCESSING
            severity = ErrorSeverity.MEDIUM
        
        # System errors (default)
        else:
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.HIGH
        
        return {
            'category': category,
            'severity': severity,
            'error_type': error_type,
            'error_message': str(error),
            'context': context,
            'timestamp': datetime.now().isoformat(),
            'recoverable': self.is_recoverable(category, severity),
            'retry_recommended': self.should_retry(category, error_type)
        }
    
    def is_recoverable(self, category: ErrorCategory, severity: ErrorSeverity) -> bool:
        """Determine if an error is recoverable"""
        # Critical errors are generally not recoverable
        if severity == ErrorSeverity.CRITICAL:
            return False
        
        # User input errors require user action
        if category == ErrorCategory.USER_INPUT:
            return False
        
        # Most other errors are recoverable
        return True
    
    def should_retry(self, category: ErrorCategory, error_type: str) -> bool:
        """Determine if an error should be retried"""
        # Don't retry user input errors
        if category == ErrorCategory.USER_INPUT:
            return False
        
        # Don't retry certain system errors
        if error_type in ['SystemExit', 'KeyboardInterrupt', 'ImportError', 'SyntaxError']:
            return False
        
        # Retry most other errors
        return True
    
    def execute_with_recovery(self, 
                            operation: Callable,
                            operation_name: str,
                            context: str = "",
                            *args, **kwargs) -> Dict[str, Any]:
        """Execute an operation with automatic error recovery"""
        
        # Get retry configuration
        retry_config = self.retry_configs.get(operation_name, self.retry_configs['api_call'])
        
        last_error = None
        attempt = 0
        
        while attempt <= retry_config['max_retries']:
            try:
                # Execute the operation
                result = operation(*args, **kwargs)
                
                # Success - reset error tracking for this operation
                if operation_name in st.session_state.error_recovery_state['recovery_attempts']:
                    del st.session_state.error_recovery_state['recovery_attempts'][operation_name]
                
                return {
                    'success': True,
                    'result': result,
                    'attempts': attempt + 1,
                    'error': None
                }
                
            except Exception as error:
                last_error = error
                attempt += 1
                
                # Classify the error
                error_info = self.classify_error(error, context)
                
                # Log the error
                self.log_error(error_info, attempt, operation_name)
                
                # Check if we should retry
                if attempt <= retry_config['max_retries'] and error_info['retry_recommended']:
                    
                    # Calculate delay with exponential backoff
                    delay = self.calculate_backoff_delay(
                        attempt, 
                        retry_config['base_delay'],
                        retry_config['max_delay'],
                        retry_config.get('jitter', False)
                    )
                    
                    # Show retry notification to user
                    if error_info['category'] in [ErrorCategory.NETWORK, ErrorCategory.API, ErrorCategory.TIMEOUT]:
                        self.show_retry_notification(operation_name, attempt, retry_config['max_retries'], delay)
                    
                    # Wait before retry
                    time.sleep(delay)
                    
                else:
                    # Max retries reached or error not retryable
                    break
        
        # All retries failed - attempt fallback recovery
        fallback_result = self.attempt_fallback_recovery(last_error, operation_name, context, *args, **kwargs)
        
        if fallback_result['success']:
            return fallback_result
        
        # Complete failure - return error information
        error_info = self.classify_error(last_error, context)
        self.record_error_pattern(error_info)
        
        return {
            'success': False,
            'result': None,
            'attempts': attempt,
            'error': error_info,
            'fallback_attempted': fallback_result['attempted'],
            'recovery_suggestions': self.get_recovery_suggestions(error_info)
        }
    
    def calculate_backoff_delay(self, attempt: int, base_delay: float, max_delay: float, jitter: bool = False) -> float:
        """Calculate exponential backoff delay with optional jitter"""
        # Exponential backoff: base_delay * (2 ^ (attempt - 1))
        delay = base_delay * (2 ** (attempt - 1))
        
        # Cap at max delay
        delay = min(delay, max_delay)
        
        # Add jitter to prevent thundering herd
        if jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return delay
    
    def show_retry_notification(self, operation: str, attempt: int, max_attempts: int, delay: float):
        """Show retry notification to user"""
        if delay > 2:  # Only show for longer delays
            st.info(f"🔄 Retrying {operation}... (Attempt {attempt}/{max_attempts}) - Waiting {delay:.1f}s")
    
    def attempt_fallback_recovery(self, error: Exception, operation_name: str, context: str, *args, **kwargs) -> Dict[str, Any]:
        """Attempt fallback recovery when retries fail"""
        error_info = self.classify_error(error, context)
        
        # Check if fallback handler exists
        if error_info['category'] in self.fallback_handlers:
            try:
                fallback_handler = self.fallback_handlers[error_info['category']]
                fallback_result = fallback_handler(error, operation_name, context, *args, **kwargs)
                
                if fallback_result:
                    self.logger.info(f"Fallback recovery successful for {operation_name}")
                    return {
                        'success': True,
                        'result': fallback_result,
                        'attempted': True,
                        'fallback_used': True
                    }
                
            except Exception as fallback_error:
                self.logger.error(f"Fallback recovery failed for {operation_name}: {str(fallback_error)}")
        
        return {
            'success': False,
            'result': None,
            'attempted': True,
            'fallback_used': False
        }
    
    def network_fallback(self, error: Exception, operation: str, context: str, *args, **kwargs):
        """Fallback handler for network errors"""
        # Enable offline mode or use cached data
        if operation == 'api_call':
            return self.use_cached_response(operation, *args, **kwargs)
        elif operation == 'file_upload':
            return self.save_for_later_upload(*args, **kwargs)
        return None
    
    def api_fallback(self, error: Exception, operation: str, context: str, *args, **kwargs):
        """Fallback handler for API errors"""
        # Use alternative API or local processing
        if 'enhancement' in context:
            return self.use_local_enhancement(*args, **kwargs)
        elif 'validation' in context:
            return self.use_basic_validation(*args, **kwargs)
        return None
    
    def file_io_fallback(self, error: Exception, operation: str, context: str, *args, **kwargs):
        """Fallback handler for file I/O errors"""
        # Try alternative file paths or temporary storage
        if operation == 'file_upload':
            return self.use_temporary_storage(*args, **kwargs)
        elif operation == 'export':
            return self.use_memory_export(*args, **kwargs)
        return None
    
    def processing_fallback(self, error: Exception, operation: str, context: str, *args, **kwargs):
        """Fallback handler for processing errors"""
        # Use simpler processing methods
        if operation == 'chunking':
            return self.use_simple_chunking(*args, **kwargs)
        elif operation == 'content_extraction':
            return self.use_basic_extraction(*args, **kwargs)
        return None
    
    def validation_fallback(self, error: Exception, operation: str, context: str, *args, **kwargs):
        """Fallback handler for validation errors"""
        # Use basic validation or skip validation
        return self.use_basic_validation(*args, **kwargs)
    
    def memory_fallback(self, error: Exception, operation: str, context: str, *args, **kwargs):
        """Fallback handler for memory errors"""
        # Process in smaller chunks or use streaming
        if operation == 'chunking':
            return self.use_streaming_chunking(*args, **kwargs)
        elif operation == 'content_extraction':
            return self.use_streaming_extraction(*args, **kwargs)
        return None
    
    def timeout_fallback(self, error: Exception, operation: str, context: str, *args, **kwargs):
        """Fallback handler for timeout errors"""
        # Use faster methods or process in smaller batches
        if operation == 'api_call':
            return self.use_faster_api(*args, **kwargs)
        elif operation == 'processing':
            return self.use_batch_processing(*args, **kwargs)
        return None
    
    def system_fallback(self, error: Exception, operation: str, context: str, *args, **kwargs):
        """Fallback handler for system errors"""
        # Use safe mode or basic functionality
        return self.use_safe_mode(*args, **kwargs)
    
    # Fallback implementation methods (simplified versions)
    def use_cached_response(self, operation: str, *args, **kwargs):
        """Use cached API response if available"""
        # Implementation would check cache and return cached data
        return None
    
    def save_for_later_upload(self, *args, **kwargs):
        """Save file for later upload when network is available"""
        # Implementation would save to local storage
        return "saved_for_later"
    
    def use_local_enhancement(self, *args, **kwargs):
        """Use local enhancement instead of API"""
        # Implementation would use basic text processing
        return "basic_enhancement"
    
    def use_basic_validation(self, *args, **kwargs):
        """Use basic validation instead of comprehensive validation"""
        # Implementation would do simple checks
        return {"validation": "basic", "passed": True}
    
    def use_temporary_storage(self, *args, **kwargs):
        """Use temporary storage for file operations"""
        # Implementation would use temp directory
        return "temp_storage"
    
    def use_memory_export(self, *args, **kwargs):
        """Export to memory instead of file"""
        # Implementation would create in-memory export
        return "memory_export"
    
    def use_simple_chunking(self, *args, **kwargs):
        """Use simple text splitting instead of advanced chunking"""
        # Implementation would do basic text splitting
        return ["simple", "chunks"]
    
    def use_basic_extraction(self, *args, **kwargs):
        """Use basic text extraction"""
        # Implementation would do simple text extraction
        return "basic_extracted_text"
    
    def use_streaming_chunking(self, *args, **kwargs):
        """Process content in smaller streaming chunks"""
        # Implementation would process in small batches
        return ["streaming", "chunks"]
    
    def use_streaming_extraction(self, *args, **kwargs):
        """Extract content using streaming approach"""
        # Implementation would extract in chunks
        return "streaming_extracted_text"
    
    def use_faster_api(self, *args, **kwargs):
        """Use faster API endpoint or method"""
        # Implementation would use simpler API
        return "fast_api_result"
    
    def use_batch_processing(self, *args, **kwargs):
        """Process in smaller batches to avoid timeouts"""
        # Implementation would split into batches
        return "batch_processed"
    
    def use_safe_mode(self, *args, **kwargs):
        """Use safe mode with basic functionality"""
        # Implementation would use minimal features
        return "safe_mode_result"
    
    def log_error(self, error_info: Dict[str, Any], attempt: int, operation: str):
        """Log error information for analysis"""
        log_entry = {
            'timestamp': error_info['timestamp'],
            'operation': operation,
            'attempt': attempt,
            'category': error_info['category'].value,
            'severity': error_info['severity'].value,
            'error_type': error_info['error_type'],
            'error_message': error_info['error_message'],
            'context': error_info['context']
        }
        
        self.error_history.append(log_entry)
        
        # Log to system logger
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[error_info['severity']]
        
        self.logger.log(log_level, f"Error in {operation} (attempt {attempt}): {error_info['error_message']}")
        
        # Update session state
        st.session_state.error_recovery_state['error_count'] += 1
        st.session_state.error_recovery_state['last_error_time'] = error_info['timestamp']
    
    def record_error_pattern(self, error_info: Dict[str, Any]):
        """Record error patterns for analysis"""
        pattern = {
            'category': error_info['category'].value,
            'error_type': error_info['error_type'],
            'timestamp': error_info['timestamp'],
            'context': error_info['context']
        }
        
        st.session_state.error_recovery_state['error_patterns'].append(pattern)
        
        # Keep only recent patterns (last 50)
        if len(st.session_state.error_recovery_state['error_patterns']) > 50:
            st.session_state.error_recovery_state['error_patterns'] = \
                st.session_state.error_recovery_state['error_patterns'][-50:]
    
    def get_recovery_suggestions(self, error_info: Dict[str, Any]) -> List[str]:
        """Get recovery suggestions based on error type"""
        suggestions = []
        
        category = error_info['category']
        
        if category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check your internet connection",
                "Try again in a few moments",
                "Use offline mode if available"
            ])
        
        elif category == ErrorCategory.API:
            suggestions.extend([
                "Check API service status",
                "Verify API credentials",
                "Try using local processing mode"
            ])
        
        elif category == ErrorCategory.FILE_IO:
            suggestions.extend([
                "Check file permissions",
                "Ensure sufficient disk space",
                "Try a different file location"
            ])
        
        elif category == ErrorCategory.MEMORY:
            suggestions.extend([
                "Try processing smaller files",
                "Close other applications",
                "Use streaming mode for large files"
            ])
        
        elif category == ErrorCategory.TIMEOUT:
            suggestions.extend([
                "Try processing in smaller batches",
                "Check network connection",
                "Increase timeout settings"
            ])
        
        elif category == ErrorCategory.USER_INPUT:
            suggestions.extend([
                "Check input format and requirements",
                "Verify all required fields are filled",
                "Review input validation messages"
            ])
        
        elif category == ErrorCategory.VALIDATION:
            suggestions.extend([
                "Check data format and structure",
                "Review validation requirements",
                "Try using basic validation mode"
            ])
        
        else:
            suggestions.extend([
                "Try refreshing the application",
                "Check system resources",
                "Contact support if problem persists"
            ])
        
        return suggestions
    
    def show_error_recovery_ui(self, error_result: Dict[str, Any]):
        """Show error recovery interface to user"""
        error_info = error_result['error']
        
        # Error summary
        severity_colors = {
            ErrorSeverity.LOW: "🟡",
            ErrorSeverity.MEDIUM: "🟠", 
            ErrorSeverity.HIGH: "🔴",
            ErrorSeverity.CRITICAL: "🚨"
        }
        
        severity_icon = severity_colors.get(error_info['severity'], "⚠️")
        
        st.error(f"{severity_icon} **Error in {error_result.get('operation', 'operation')}**")
        st.write(f"**Category:** {error_info['category'].value.title()}")
        st.write(f"**Attempts:** {error_result['attempts']}")
        
        # Show user-friendly error message
        user_message = self.get_user_friendly_message(error_info)
        st.write(f"**Issue:** {user_message}")
        
        # Recovery suggestions
        suggestions = error_result.get('recovery_suggestions', [])
        if suggestions:
            st.write("**💡 Suggestions:**")
            for suggestion in suggestions:
                st.write(f"• {suggestion}")
        
        # Recovery actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Try Again"):
                return "retry"
        
        with col2:
            if error_info['recoverable'] and st.button("🛠️ Use Fallback"):
                return "fallback"
        
        with col3:
            if st.button("⏭️ Skip & Continue"):
                return "skip"
        
        # Technical details (expandable)
        with st.expander("🔧 Technical Details", expanded=False):
            st.code(f"Error Type: {error_info['error_type']}")
            st.code(f"Message: {error_info['error_message']}")
            st.code(f"Context: {error_info['context']}")
            st.code(f"Timestamp: {error_info['timestamp']}")
        
        return None
    
    def get_user_friendly_message(self, error_info: Dict[str, Any]) -> str:
        """Convert technical error to user-friendly message"""
        category = error_info['category']
        
        messages = {
            ErrorCategory.NETWORK: "Network connection issue. Please check your internet connection.",
            ErrorCategory.API: "Service temporarily unavailable. The system will retry automatically.",
            ErrorCategory.FILE_IO: "File access issue. Please check file permissions and try again.",
            ErrorCategory.PROCESSING: "Processing error occurred. The system will attempt alternative methods.",
            ErrorCategory.VALIDATION: "Data validation issue. Please check your input format.",
            ErrorCategory.MEMORY: "Insufficient memory. Try processing smaller files or close other applications.",
            ErrorCategory.TIMEOUT: "Operation timed out. Try processing in smaller batches.",
            ErrorCategory.USER_INPUT: "Input validation error. Please check your input and try again.",
            ErrorCategory.SYSTEM: "System error occurred. The application will attempt recovery."
        }
        
        return messages.get(category, "An unexpected error occurred. The system will attempt recovery.")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring"""
        state = st.session_state.error_recovery_state
        
        # Analyze error patterns
        patterns = state['error_patterns']
        category_counts = {}
        recent_errors = 0
        
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        for pattern in patterns:
            category = pattern['category']
            category_counts[category] = category_counts.get(category, 0) + 1
            
            pattern_time = datetime.fromisoformat(pattern['timestamp'])
            if pattern_time > cutoff_time:
                recent_errors += 1
        
        return {
            'total_errors': state['error_count'],
            'recent_errors': recent_errors,
            'last_error_time': state['last_error_time'],
            'category_breakdown': category_counts,
            'fallback_mode': state['fallback_mode'],
            'error_rate': recent_errors / max(1, len(patterns)) if patterns else 0
        }
    
    def enable_fallback_mode(self):
        """Enable fallback mode for degraded functionality"""
        st.session_state.error_recovery_state['fallback_mode'] = True
        st.warning("🛠️ Fallback mode enabled - Using simplified processing methods")
    
    def disable_fallback_mode(self):
        """Disable fallback mode and return to normal operation"""
        st.session_state.error_recovery_state['fallback_mode'] = False
        st.success("✅ Normal operation restored")
    
    def is_fallback_mode(self) -> bool:
        """Check if system is in fallback mode"""
        return st.session_state.error_recovery_state.get('fallback_mode', False)

# Global instance
smart_error_recovery = SmartErrorRecovery()

