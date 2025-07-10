"""
Enhanced Logging Module
======================

Provides comprehensive logging capabilities for the Enhanced Universal AI Training Data Creator.
Includes detailed traces, metadata logging, audit trails, and cost tracking.

Features:
- Structured logging with metadata
- Cost and usage tracking
- Audit trail for compliance
- Performance logging
- Error tracking with context
- Log viewer integration
"""

import logging
import json
import time
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import streamlit as st
from pathlib import Path

# Configure logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


class EnhancedLogging:
    """Enhanced logging system with structured logging and metadata tracking"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup main application logger
        self.logger = logging.getLogger("enhanced_ai_trainer")
        self.logger.setLevel(logging.INFO)
        
        # Setup file handler
        log_file = self.log_dir / "app.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers if not already added
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        # Initialize tracking
        self.session_id = self._generate_session_id()
        self.cost_tracker = CostTracker()
        self.audit_logger = AuditLogger(self.log_dir)
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{id(self)}"
    
    def log_file_upload(self, filename: str, file_size: int, file_type: str, file_hash: str):
        """Log file upload with metadata"""
        metadata = {
            'session_id': self.session_id,
            'event_type': 'file_upload',
            'filename': filename,
            'file_size': file_size,
            'file_type': file_type,
            'file_hash': file_hash,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"FILE_UPLOAD: {filename} ({file_size} bytes, {file_type})")
        self.audit_logger.log_event('file_upload', metadata)
    
    def log_content_extraction(self, file_type: str, extraction_method: str, 
                             content_length: int, extraction_time: float, success: bool):
        """Log content extraction with performance metrics"""
        metadata = {
            'session_id': self.session_id,
            'event_type': 'content_extraction',
            'file_type': file_type,
            'extraction_method': extraction_method,
            'content_length': content_length,
            'extraction_time': extraction_time,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"EXTRACTION_{status}: {file_type} via {extraction_method} "
                        f"({content_length} chars in {extraction_time:.3f}s)")
        
        self.audit_logger.log_event('content_extraction', metadata)
    
    def log_content_detection(self, content_type: str, confidence: float, 
                            detection_method: str, content_stats: Dict[str, Any]):
        """Log content type detection with confidence metrics"""
        metadata = {
            'session_id': self.session_id,
            'event_type': 'content_detection',
            'detected_type': content_type,
            'confidence': confidence,
            'detection_method': detection_method,
            'content_stats': content_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"DETECTION: {content_type} (confidence: {confidence:.3f}) "
                        f"via {detection_method}")
        
        self.audit_logger.log_event('content_detection', metadata)
    
    def log_ai_enhancement(self, tone: str, model: str, input_tokens: int, 
                          output_tokens: int, cost: float, enhancement_time: float):
        """Log AI enhancement with cost and performance tracking"""
        metadata = {
            'session_id': self.session_id,
            'event_type': 'ai_enhancement',
            'tone': tone,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens,
            'cost_usd': cost,
            'enhancement_time': enhancement_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"AI_ENHANCEMENT: {tone} tone via {model} "
                        f"({input_tokens}→{output_tokens} tokens, ${cost:.4f}, {enhancement_time:.3f}s)")
        
        # Track costs
        self.cost_tracker.add_cost(cost, model, input_tokens + output_tokens)
        
        self.audit_logger.log_event('ai_enhancement', metadata)
    
    def log_quality_analysis(self, quality_scores: Dict[str, float], 
                           quality_flags: List[str], passed_threshold: bool):
        """Log quality analysis results"""
        metadata = {
            'session_id': self.session_id,
            'event_type': 'quality_analysis',
            'quality_scores': quality_scores,
            'quality_flags': quality_flags,
            'passed_threshold': passed_threshold,
            'overall_score': sum(quality_scores.values()) / len(quality_scores) if quality_scores else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        overall_score = metadata['overall_score']
        status = "PASSED" if passed_threshold else "FAILED"
        
        self.logger.info(f"QUALITY_{status}: Overall score {overall_score:.3f}, "
                        f"flags: {', '.join(quality_flags) if quality_flags else 'none'}")
        
        self.audit_logger.log_event('quality_analysis', metadata)
    
    def log_manual_review(self, reviewed_items: int, approved_items: int, 
                         rejected_items: int, review_time: float):
        """Log manual review session"""
        metadata = {
            'session_id': self.session_id,
            'event_type': 'manual_review',
            'reviewed_items': reviewed_items,
            'approved_items': approved_items,
            'rejected_items': rejected_items,
            'approval_rate': approved_items / reviewed_items if reviewed_items > 0 else 0,
            'review_time': review_time,
            'timestamp': datetime.now().isoformat()
        }
        
        approval_rate = metadata['approval_rate'] * 100
        self.logger.info(f"MANUAL_REVIEW: {reviewed_items} items reviewed, "
                        f"{approved_items} approved ({approval_rate:.1f}%) in {review_time:.1f}s")
        
        self.audit_logger.log_event('manual_review', metadata)
    
    def log_export(self, export_format: str, item_count: int, file_size: int, 
                  export_destination: str, export_time: float):
        """Log data export with metadata"""
        metadata = {
            'session_id': self.session_id,
            'event_type': 'data_export',
            'export_format': export_format,
            'item_count': item_count,
            'file_size': file_size,
            'export_destination': export_destination,
            'export_time': export_time,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"EXPORT: {item_count} items to {export_format} "
                        f"({file_size} bytes) → {export_destination} in {export_time:.3f}s")
        
        self.audit_logger.log_event('data_export', metadata)
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log error with context"""
        metadata = {
            'session_id': self.session_id,
            'event_type': 'error',
            'error_type': error_type,
            'error_message': error_message,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.error(f"ERROR_{error_type}: {error_message}")
        if context:
            self.logger.error(f"ERROR_CONTEXT: {json.dumps(context, indent=2)}")
        
        self.audit_logger.log_event('error', metadata)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        # Safe session_id parsing
        try:
            session_parts = self.session_id.split('_')
            if len(session_parts) >= 3:
                session_start = f"{session_parts[1]}_{session_parts[2]}"
            else:
                session_start = self.session_id
        except (AttributeError, IndexError):
            session_start = "unknown"
            
        return {
            'session_id': self.session_id,
            'total_cost': self.cost_tracker.get_total_cost(),
            'total_tokens': self.cost_tracker.get_total_tokens(),
            'cost_by_model': self.cost_tracker.get_cost_by_model(),
            'session_start': session_start
        }
    
    def render_log_viewer(self):
        """Render log viewer in Streamlit"""
        st.markdown("### 📝 Application Logs")
        
        # Session summary
        summary = self.get_session_summary()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Session Cost", f"${summary['total_cost']:.4f}")
        
        with col2:
            st.metric("Total Tokens", f"{summary['total_tokens']:,}")
        
        with col3:
            st.metric("Session ID", summary['session_id'][-8:])
        
        # Cost breakdown
        if summary['cost_by_model']:
            st.markdown("**Cost by Model:**")
            for model, cost in summary['cost_by_model'].items():
                st.write(f"- {model}: ${cost:.4f}")
        
        # Recent logs
        st.markdown("**Recent Log Entries:**")
        
        try:
            log_file = self.log_dir / "app.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                # Show last 20 lines
                recent_lines = lines[-20:] if len(lines) > 20 else lines
                
                for line in reversed(recent_lines):
                    if line.strip():
                        # Color code by log level
                        if "ERROR" in line:
                            st.error(line.strip())
                        elif "WARNING" in line:
                            st.warning(line.strip())
                        elif "INFO" in line:
                            st.info(line.strip())
                        else:
                            st.text(line.strip())
            else:
                st.info("No log file found yet.")
                
        except Exception as e:
            st.error(f"Error reading log file: {e}")


class CostTracker:
    """Track API costs and usage"""
    
    def __init__(self):
        self.costs = []
        self.total_cost = 0.0
        self.total_tokens = 0
        self.cost_by_model = {}
    
    def add_cost(self, cost: float, model: str, tokens: int):
        """Add cost entry"""
        self.costs.append({
            'cost': cost,
            'model': model,
            'tokens': tokens,
            'timestamp': datetime.now().isoformat()
        })
        
        self.total_cost += cost
        self.total_tokens += tokens
        
        if model not in self.cost_by_model:
            self.cost_by_model[model] = 0.0
        self.cost_by_model[model] += cost
    
    def get_total_cost(self) -> float:
        return self.total_cost
    
    def get_total_tokens(self) -> int:
        return self.total_tokens
    
    def get_cost_by_model(self) -> Dict[str, float]:
        return self.cost_by_model.copy()


class AuditLogger:
    """Audit trail logger for compliance"""
    
    def __init__(self, log_dir: Path):
        self.audit_file = log_dir / "audit.jsonl"
    
    def log_event(self, event_type: str, metadata: Dict[str, Any]):
        """Log audit event"""
        audit_entry = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            **metadata
        }
        
        try:
            with open(self.audit_file, 'a') as f:
                f.write(json.dumps(audit_entry) + '\n')
        except Exception as e:
            logging.error(f"Failed to write audit log: {e}")


# Global enhanced logging instance
enhanced_logging = EnhancedLogging()