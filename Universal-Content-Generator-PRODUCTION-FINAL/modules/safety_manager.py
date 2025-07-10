#!/usr/bin/env python3
"""
Safety Manager Module
Handles auto-save, session recovery, checkpoints, and error handling
"""

import os
import pickle
import time
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import streamlit as st
import logging

class SafetyManager:
    def __init__(self):
        self.auto_save_interval = 30  # seconds
        self.max_file_size = 50_000_000  # 50MB
        self.checkpoint_frequency = 5  # every 5 actions
        self.sessions_dir = "sessions"
        self.checkpoints_dir = "checkpoints"
        self.logs_dir = "logs"
        
        # Create directories
        for dir_path in [self.sessions_dir, self.checkpoints_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize session tracking
        if 'last_auto_save' not in st.session_state:
            st.session_state.last_auto_save = time.time()
        if 'action_count' not in st.session_state:
            st.session_state.action_count = 0
        if 'session_id' not in st.session_state:
            st.session_state.session_id = self.generate_session_id()
    
    def setup_logging(self):
        """Set up comprehensive logging system"""
        log_file = os.path.join(self.logs_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() % 10000)}"
    
    def validate_file(self, file) -> Dict[str, Any]:
        """Pre-processing file validation with warnings and recommendations"""
        try:
            file_size = len(file.getvalue()) if hasattr(file, 'getvalue') else 0
            file_type = getattr(file, 'type', 'unknown')
            file_name = getattr(file, 'name', 'unknown')
            
            analysis = {
                'file_name': file_name,
                'file_size': file_size,
                'file_type': file_type,
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'recommendations': [],
                'estimated_processing_time': self.estimate_processing_time(file_size)
            }
            
            # Size validation
            if file_size == 0:
                analysis['errors'].append("File appears to be empty")
                analysis['is_valid'] = False
            elif file_size > self.max_file_size:
                analysis['warnings'].append(f"Large file ({self.format_file_size(file_size)}) - may take longer to process")
                analysis['recommendations'].append("Consider splitting into smaller sections for faster processing")
            
            # Format validation
            supported_formats = ['application/pdf', 'text/plain', 'text/markdown', 
                               'application/vnd.openxmlformats-officedocument.wordprocessingml.document']
            
            if file_type not in supported_formats:
                analysis['warnings'].append(f"File type '{file_type}' may not be fully supported")
                analysis['recommendations'].append("Convert to PDF, TXT, DOCX, or MD for best results")
            
            # Processing time warnings
            if analysis['estimated_processing_time'] > 300:  # 5 minutes
                analysis['warnings'].append("Large file may take several minutes to process")
                analysis['recommendations'].append("Consider processing during a break or in smaller chunks")
            
            self.logger.info(f"File validation completed: {file_name} ({self.format_file_size(file_size)})")
            return analysis
            
        except Exception as e:
            self.logger.error(f"File validation error: {str(e)}")
            return {
                'is_valid': False,
                'errors': [f"File validation failed: {str(e)}"],
                'warnings': [],
                'recommendations': ["Try uploading the file again or use a different format"]
            }
    
    def estimate_processing_time(self, file_size: int) -> int:
        """Estimate processing time based on file size"""
        # Rough estimation: 1MB = 10 seconds processing time
        base_time = (file_size / 1_000_000) * 10
        return max(5, int(base_time))  # Minimum 5 seconds
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes/1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes/(1024**2):.1f} MB"
        else:
            return f"{size_bytes/(1024**3):.1f} GB"
    
    def auto_save_session(self, force: bool = False):
        """Continuous auto-save functionality"""
        current_time = time.time()
        
        if force or (current_time - st.session_state.last_auto_save) >= self.auto_save_interval:
            try:
                session_data = {
                    'session_id': st.session_state.session_id,
                    'timestamp': datetime.now().isoformat(),
                    'uploaded_file_name': st.session_state.get('uploaded_file_name'),
                    'processed_chunks': st.session_state.get('processed_chunks', []),
                    'validated_data': st.session_state.get('validated_data', {}),
                    'enhanced_content': st.session_state.get('enhanced_content', []),
                    'user_settings': st.session_state.get('user_settings', {}),
                    'current_step': st.session_state.get('current_step', 'upload'),
                    'processing_progress': st.session_state.get('processing_progress', 0)
                }
                
                session_file = os.path.join(self.sessions_dir, f"{st.session_state.session_id}.pkl")
                with open(session_file, 'wb') as f:
                    pickle.dump(session_data, f)
                
                st.session_state.last_auto_save = current_time
                self.logger.info(f"Auto-save completed: {st.session_state.session_id}")
                
                # Clean up old sessions (older than 7 days)
                self.cleanup_old_sessions()
                
            except Exception as e:
                self.logger.error(f"Auto-save failed: {str(e)}")
    
    def get_recent_sessions(self) -> List[Dict[str, Any]]:
        """Get list of recent sessions for recovery"""
        sessions = []
        
        try:
            if os.path.exists(self.sessions_dir):
                for file_name in os.listdir(self.sessions_dir):
                    if file_name.endswith('.pkl'):
                        file_path = os.path.join(self.sessions_dir, file_name)
                        try:
                            with open(file_path, 'rb') as f:
                                session_data = pickle.load(f)
                                sessions.append({
                                    'file_path': file_path,
                                    'session_id': session_data.get('session_id', 'unknown'),
                                    'timestamp': session_data.get('timestamp', ''),
                                    'uploaded_file': session_data.get('uploaded_file_name', 'No file'),
                                    'chunks_count': len(session_data.get('processed_chunks', [])),
                                    'current_step': session_data.get('current_step', 'unknown')
                                })
                        except Exception as e:
                            self.logger.warning(f"Could not load session {file_name}: {str(e)}")
            
            # Sort by timestamp (newest first)
            sessions.sort(key=lambda x: x['timestamp'], reverse=True)
            return sessions[:5]  # Return last 5 sessions
            
        except Exception as e:
            self.logger.error(f"Error getting recent sessions: {str(e)}")
            return []
    
    def restore_session(self, session_file_path: str) -> bool:
        """Restore session from file"""
        try:
            with open(session_file_path, 'rb') as f:
                session_data = pickle.load(f)
            
            # Restore session state
            for key, value in session_data.items():
                if key != 'timestamp':  # Don't restore timestamp
                    st.session_state[key] = value
            
            self.logger.info(f"Session restored: {session_data.get('session_id', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Session restoration failed: {str(e)}")
            return False
    
    def create_checkpoint(self, step_name: str, data: Dict[str, Any]):
        """Create workflow checkpoint"""
        try:
            checkpoint = {
                'step': step_name,
                'session_id': st.session_state.session_id,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'user_settings': st.session_state.get('user_settings', {}),
                'action_count': st.session_state.action_count
            }
            
            checkpoint_file = os.path.join(
                self.checkpoints_dir, 
                f"{st.session_state.session_id}_{step_name}_{datetime.now().strftime('%H%M%S')}.pkl"
            )
            
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint, f)
            
            self.logger.info(f"Checkpoint created: {step_name}")
            return checkpoint_file
            
        except Exception as e:
            self.logger.error(f"Checkpoint creation failed: {str(e)}")
            return None
    
    def get_checkpoints(self) -> List[Dict[str, Any]]:
        """Get available checkpoints for current session"""
        checkpoints = []
        
        try:
            if os.path.exists(self.checkpoints_dir):
                session_prefix = st.session_state.session_id
                for file_name in os.listdir(self.checkpoints_dir):
                    if file_name.startswith(session_prefix) and file_name.endswith('.pkl'):
                        file_path = os.path.join(self.checkpoints_dir, file_name)
                        try:
                            with open(file_path, 'rb') as f:
                                checkpoint_data = pickle.load(f)
                                checkpoints.append({
                                    'file_path': file_path,
                                    'step': checkpoint_data.get('step', 'unknown'),
                                    'timestamp': checkpoint_data.get('timestamp', ''),
                                    'action_count': checkpoint_data.get('action_count', 0)
                                })
                        except Exception as e:
                            self.logger.warning(f"Could not load checkpoint {file_name}: {str(e)}")
            
            # Sort by action count (newest first)
            checkpoints.sort(key=lambda x: x['action_count'], reverse=True)
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"Error getting checkpoints: {str(e)}")
            return []
    
    def restore_checkpoint(self, checkpoint_file_path: str) -> bool:
        """Restore from checkpoint"""
        try:
            with open(checkpoint_file_path, 'rb') as f:
                checkpoint_data = pickle.load(f)
            
            # Restore data
            for key, value in checkpoint_data['data'].items():
                st.session_state[key] = value
            
            # Restore settings
            st.session_state.user_settings = checkpoint_data.get('user_settings', {})
            
            self.logger.info(f"Checkpoint restored: {checkpoint_data.get('step', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Checkpoint restoration failed: {str(e)}")
            return False
    
    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """Smart error recovery with user-friendly messages"""
        error_type = type(error).__name__
        error_message = str(error)
        
        self.logger.error(f"Error in {context}: {error_type} - {error_message}")
        
        # Auto-save before handling error
        self.auto_save_session(force=True)
        
        recovery_options = {
            'error_type': error_type,
            'user_message': '',
            'technical_message': error_message,
            'recovery_actions': [],
            'fallback_available': False
        }
        
        # Handle specific error types
        if 'MemoryError' in error_type or 'memory' in error_message.lower():
            recovery_options.update({
                'user_message': "⚠️ Memory limit reached. The file might be too large for processing.",
                'recovery_actions': [
                    "Try processing a smaller file",
                    "Split the content into smaller sections",
                    "Restart the application to free memory"
                ],
                'fallback_available': True
            })
        
        elif 'TimeoutError' in error_type or 'timeout' in error_message.lower():
            recovery_options.update({
                'user_message': "⏱️ Processing timeout. The operation took too long.",
                'recovery_actions': [
                    "Try again with a smaller file",
                    "Check your internet connection",
                    "Resume from the last checkpoint"
                ],
                'fallback_available': True
            })
        
        elif 'ConnectionError' in error_type or 'connection' in error_message.lower():
            recovery_options.update({
                'user_message': "🌐 Connection issue. Unable to reach external services.",
                'recovery_actions': [
                    "Check your internet connection",
                    "Try again in a few moments",
                    "Use offline processing mode"
                ],
                'fallback_available': True
            })
        
        elif 'APIError' in error_type or 'api' in error_message.lower():
            recovery_options.update({
                'user_message': "🔌 API service issue. External service is temporarily unavailable.",
                'recovery_actions': [
                    "Try again in a few minutes",
                    "Check API key configuration",
                    "Use alternative processing mode"
                ],
                'fallback_available': True
            })
        
        else:
            recovery_options.update({
                'user_message': f"❌ Unexpected error occurred: {error_type}",
                'recovery_actions': [
                    "Try the operation again",
                    "Restart the application",
                    "Contact support if the issue persists"
                ],
                'fallback_available': False
            })
        
        return recovery_options
    
    def cleanup_old_sessions(self):
        """Clean up sessions older than 7 days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for dir_path in [self.sessions_dir, self.checkpoints_dir]:
                if os.path.exists(dir_path):
                    for file_name in os.listdir(dir_path):
                        file_path = os.path.join(dir_path, file_name)
                        if os.path.isfile(file_path):
                            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                            if file_time < cutoff_date:
                                os.remove(file_path)
                                self.logger.info(f"Cleaned up old file: {file_name}")
        
        except Exception as e:
            self.logger.warning(f"Cleanup failed: {str(e)}")
    
    def show_recovery_dialog(self):
        """Show session recovery dialog in Streamlit"""
        recent_sessions = self.get_recent_sessions()
        
        if recent_sessions and st.session_state.get('show_recovery', True):
            st.info("🔄 Previous sessions found!")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                session_options = [
                    f"{s['uploaded_file']} - {s['current_step']} ({s['chunks_count']} chunks) - {s['timestamp'][:16]}"
                    for s in recent_sessions
                ]
                selected_idx = st.selectbox(
                    "Select session to restore:",
                    range(len(session_options)),
                    format_func=lambda x: session_options[x]
                )
            
            with col2:
                if st.button("🔄 Resume Session"):
                    if self.restore_session(recent_sessions[selected_idx]['file_path']):
                        st.success("✅ Session restored successfully!")
                        st.session_state.show_recovery = False
                        st.rerun()
                    else:
                        st.error("❌ Failed to restore session")
            
            with col3:
                if st.button("🆕 Start Fresh"):
                    st.session_state.show_recovery = False
                    st.session_state.session_id = self.generate_session_id()
                    st.rerun()
    
    def increment_action_count(self):
        """Increment action count and create checkpoint if needed"""
        st.session_state.action_count += 1
        
        if st.session_state.action_count % self.checkpoint_frequency == 0:
            # Create automatic checkpoint
            checkpoint_data = {
                'processed_chunks': st.session_state.get('processed_chunks', []),
                'validated_data': st.session_state.get('validated_data', {}),
                'enhanced_content': st.session_state.get('enhanced_content', []),
                'current_step': st.session_state.get('current_step', 'upload')
            }
            self.create_checkpoint(f"auto_checkpoint_{st.session_state.action_count}", checkpoint_data)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for monitoring"""
        try:
            return {
                'session_id': st.session_state.session_id,
                'last_auto_save': datetime.fromtimestamp(st.session_state.last_auto_save).isoformat(),
                'action_count': st.session_state.action_count,
                'sessions_count': len(os.listdir(self.sessions_dir)) if os.path.exists(self.sessions_dir) else 0,
                'checkpoints_count': len(self.get_checkpoints()),
                'memory_usage': self.get_memory_usage(),
                'uptime': time.time() - st.session_state.get('app_start_time', time.time())
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {str(e)}")
            return {'error': str(e)}
    
    def get_memory_usage(self) -> str:
        """Get approximate memory usage"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            return f"{memory_mb:.1f} MB"
        except ImportError:
            return "Unknown (psutil not available)"
        except Exception:
            return "Unknown"

# Global instance
safety_manager = SafetyManager()

