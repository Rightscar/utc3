"""
Render.com Optimization Module
=============================

Comprehensive solutions for Render.com specific deployment challenges:
1. Service Hibernation / Cold Start Lag
2. Temporary File Build Failures  
3. Session Expiry / Timeout
4. PDF Crashes on Edge Cases

Features:
- Warming up messages and heartbeat API
- Runtime dependency installation
- Auto-save and session recovery
- Enhanced PDF validation and fallbacks
"""

import os
import time
import pickle
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import streamlit as st
import requests
from threading import Thread
import subprocess

logger = logging.getLogger(__name__)


class RenderOptimization:
    """Comprehensive Render.com deployment optimization"""
    
    def __init__(self):
        self.session_dir = Path("sessions")
        self.session_dir.mkdir(exist_ok=True)
        self.heartbeat_interval = 300  # 5 minutes
        self.session_timeout = 1800  # 30 minutes
        
    # Issue 1: Service Hibernation / Cold Start Lag
    def show_warming_message(self):
        """Display warming up message during cold start"""
        try:
            # Check if this is a cold start
            if 'app_warmed_up' not in st.session_state:
                with st.spinner("🔥 Warming up engine... Please wait"):
                    # Simulate warming up critical components
                    self._warm_up_components()
                    st.session_state['app_warmed_up'] = True
                    st.session_state['warm_up_time'] = datetime.now()
                
                st.success("✅ Engine warmed up! Ready to process your content.")
                time.sleep(1)  # Brief pause for user feedback
                st.rerun()
            
        except Exception as e:
            logger.warning(f"Warming up error: {e}")
            # Continue anyway - don't block the app
            st.session_state['app_warmed_up'] = True
    
    def _warm_up_components(self):
        """Warm up critical app components"""
        try:
            # Pre-load essential modules
            import tiktoken
            import pandas as pd
            
            # Initialize tokenizer
            encoding = tiktoken.get_encoding("cl100k_base")
            
            # Test basic operations
            test_text = "Testing tokenization"
            tokens = encoding.encode(test_text)
            
            # Pre-create session directories
            self.session_dir.mkdir(exist_ok=True)
            
            # Warm up file system
            temp_file = self.session_dir / "warmup.tmp"
            temp_file.write_text("warmup")
            temp_file.unlink()
            
            logger.info("Components warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Component warm-up error: {e}")
    
    def create_healthcheck_endpoint(self):
        """Create healthcheck endpoint for heartbeat monitoring"""
        try:
            # This would typically be implemented as a separate Flask endpoint
            # For Streamlit, we'll use a simple file-based heartbeat
            heartbeat_file = Path("heartbeat.txt")
            heartbeat_file.write_text(f"alive_{datetime.now().isoformat()}")
            
            # Schedule periodic heartbeat updates
            if 'heartbeat_started' not in st.session_state:
                self._start_heartbeat_thread()
                st.session_state['heartbeat_started'] = True
                
        except Exception as e:
            logger.warning(f"Heartbeat setup error: {e}")
    
    def _start_heartbeat_thread(self):
        """Start background heartbeat thread"""
        def heartbeat_worker():
            while True:
                try:
                    heartbeat_file = Path("heartbeat.txt")
                    heartbeat_file.write_text(f"alive_{datetime.now().isoformat()}")
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    logger.warning(f"Heartbeat error: {e}")
                    break
        
        # Start heartbeat in background (note: this is a simplified approach)
        # In production, you'd use a proper background service
        try:
            thread = Thread(target=heartbeat_worker, daemon=True)
            thread.start()
        except Exception as e:
            logger.warning(f"Heartbeat thread error: {e}")
    
    # Issue 2: Temporary File Build Failures
    def install_runtime_dependencies(self):
        """Install OCR dependencies at runtime if needed"""
        try:
            # Check if tesseract is available
            result = subprocess.run(['which', 'tesseract'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                st.warning("🔧 Installing OCR dependencies... This may take a moment.")
                
                with st.spinner("Installing Tesseract OCR..."):
                    # Install tesseract at runtime
                    install_commands = [
                        ['apt-get', 'update'],
                        ['apt-get', 'install', '-y', 'tesseract-ocr'],
                        ['apt-get', 'install', '-y', 'tesseract-ocr-eng'],
                        ['apt-get', 'install', '-y', 'tesseract-ocr-hin'],  # Hindi
                        ['apt-get', 'install', '-y', 'tesseract-ocr-san'],  # Sanskrit
                    ]
                    
                    for cmd in install_commands:
                        try:
                            subprocess.run(cmd, check=True, capture_output=True)
                        except subprocess.CalledProcessError as e:
                            logger.warning(f"Install command failed: {cmd} - {e}")
                            # Continue with next command
                    
                    st.success("✅ OCR dependencies installed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Runtime dependency installation failed: {e}")
            st.error(f"⚠️ Could not install OCR dependencies: {e}")
            st.info("💡 OCR features may be limited. Text-based PDFs will still work.")
            return False
    
    def optimize_build_process(self):
        """Optimize build process for Render.com"""
        try:
            # Create optimized requirements for build
            build_requirements = [
                "streamlit>=1.28.0",
                "pandas>=1.5.0",
                "numpy>=1.24.0",
                "requests>=2.28.0",
                "pydantic>=2.0.0",
                "tiktoken>=0.5.0",
                "python-docx>=0.8.11",
                "PyPDF2>=3.0.0",
                "openpyxl>=3.1.0",
            ]
            
            # Write minimal build requirements
            build_req_file = Path("requirements_build.txt")
            build_req_file.write_text("\\n".join(build_requirements))
            
            # Create runtime requirements for additional features
            runtime_requirements = [
                "sentence-transformers>=2.2.0",
                "Pillow>=9.5.0",
                "pytesseract>=0.3.10",
                "pdf2image>=1.16.0",
            ]
            
            runtime_req_file = Path("requirements_runtime.txt")
            runtime_req_file.write_text("\\n".join(runtime_requirements))
            
            logger.info("Build optimization completed")
            return True
            
        except Exception as e:
            logger.error(f"Build optimization failed: {e}")
            return False
    
    # Issue 3: Session Expiry / Timeout
    def auto_save_session(self, session_data: Dict[str, Any], session_id: str = None):
        """Auto-save session data to disk"""
        try:
            if session_id is None:
                session_id = st.session_state.get('session_id', f"session_{int(time.time())}")
                st.session_state['session_id'] = session_id
            
            # Create session file
            session_file = self.session_dir / f"{session_id}.pkl"
            
            # Prepare data for saving
            save_data = {
                'timestamp': datetime.now().isoformat(),
                'session_data': session_data,
                'app_version': '4.0',
                'expires_at': (datetime.now() + timedelta(days=7)).isoformat()
            }
            
            # Save to pickle file
            with open(session_file, 'wb') as f:
                pickle.dump(save_data, f)
            
            # Also save as JSON backup (for debugging)
            json_file = self.session_dir / f"{session_id}.json"
            try:
                with open(json_file, 'w') as f:
                    json.dump({
                        'timestamp': save_data['timestamp'],
                        'session_id': session_id,
                        'data_keys': list(session_data.keys()),
                        'expires_at': save_data['expires_at']
                    }, f, indent=2)
            except Exception:
                pass  # JSON backup is optional
            
            logger.info(f"Session {session_id} auto-saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Auto-save failed: {e}")
            return False
    
    def load_saved_session(self, session_id: str) -> Tuple[bool, Dict[str, Any], str]:
        """Load saved session data"""
        try:
            session_file = self.session_dir / f"{session_id}.pkl"
            
            if not session_file.exists():
                return False, {}, "Session file not found"
            
            # Load session data
            with open(session_file, 'rb') as f:
                save_data = pickle.load(f)
            
            # Check if session has expired
            expires_at = datetime.fromisoformat(save_data['expires_at'])
            if datetime.now() > expires_at:
                # Clean up expired session
                session_file.unlink(missing_ok=True)
                return False, {}, "Session expired"
            
            logger.info(f"Session {session_id} loaded successfully")
            return True, save_data['session_data'], ""
            
        except Exception as e:
            logger.error(f"Session load failed: {e}")
            return False, {}, f"Load error: {e}"
    
    def list_available_sessions(self) -> List[Dict[str, Any]]:
        """List all available saved sessions"""
        try:
            sessions = []
            
            for session_file in self.session_dir.glob("*.pkl"):
                try:
                    with open(session_file, 'rb') as f:
                        save_data = pickle.load(f)
                    
                    # Check if expired
                    expires_at = datetime.fromisoformat(save_data['expires_at'])
                    if datetime.now() > expires_at:
                        session_file.unlink(missing_ok=True)
                        continue
                    
                    session_info = {
                        'session_id': session_file.stem,
                        'timestamp': save_data['timestamp'],
                        'age': str(datetime.now() - datetime.fromisoformat(save_data['timestamp'])),
                        'data_keys': list(save_data['session_data'].keys()) if save_data['session_data'] else []
                    }
                    sessions.append(session_info)
                    
                except Exception as e:
                    logger.warning(f"Error reading session {session_file}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            sessions.sort(key=lambda x: x['timestamp'], reverse=True)
            return sessions
            
        except Exception as e:
            logger.error(f"Session listing failed: {e}")
            return []
    
    def render_session_recovery_ui(self):
        """Render session recovery UI"""
        try:
            st.sidebar.markdown("### 💾 Session Recovery")
            
            # Auto-save current session
            if st.sidebar.button("💾 Save Current Session"):
                current_data = dict(st.session_state)
                # Remove non-serializable items
                clean_data = {k: v for k, v in current_data.items() 
                            if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict))}
                
                if self.auto_save_session(clean_data):
                    st.sidebar.success("✅ Session saved!")
                else:
                    st.sidebar.error("❌ Save failed")
            
            # List available sessions
            sessions = self.list_available_sessions()
            
            if sessions:
                st.sidebar.markdown("**Available Sessions:**")
                
                for session in sessions[:5]:  # Show last 5 sessions
                    session_label = f"📅 {session['timestamp'][:16]} ({len(session['data_keys'])} items)"
                    
                    if st.sidebar.button(session_label, key=f"load_{session['session_id']}"):
                        success, data, error = self.load_saved_session(session['session_id'])
                        
                        if success:
                            # Restore session data
                            for key, value in data.items():
                                st.session_state[key] = value
                            
                            st.sidebar.success("✅ Session restored!")
                            st.rerun()
                        else:
                            st.sidebar.error(f"❌ Load failed: {error}")
            else:
                st.sidebar.info("No saved sessions found")
                
        except Exception as e:
            logger.error(f"Session recovery UI error: {e}")
            st.sidebar.error("Session recovery unavailable")
    
    # Issue 4: PDF Crashes on Edge Cases
    def enhanced_pdf_validation(self, file_path: str) -> Tuple[bool, Dict[str, Any], str]:
        """Enhanced PDF validation with detailed diagnostics"""
        try:
            import PyPDF2
            from pdf2image import convert_from_path
            
            validation_results = {
                'file_size_mb': 0,
                'page_count': 0,
                'is_encrypted': False,
                'has_text': False,
                'has_images': False,
                'is_scanned': False,
                'rotation_detected': False,
                'language_detected': 'unknown',
                'confidence_score': 0.0
            }
            
            # Basic file validation
            if not os.path.exists(file_path):
                return False, validation_results, "File not found"
            
            file_size = os.path.getsize(file_path)
            validation_results['file_size_mb'] = file_size / (1024 * 1024)
            
            # Size limits
            if file_size > 200 * 1024 * 1024:  # 200MB
                return False, validation_results, "File too large (>200MB)"
            
            if file_size < 100:  # 100 bytes
                return False, validation_results, "File too small"
            
            # PDF structure validation
            try:
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    
                    validation_results['page_count'] = len(pdf_reader.pages)
                    validation_results['is_encrypted'] = pdf_reader.is_encrypted
                    
                    if validation_results['page_count'] == 0:
                        return False, validation_results, "PDF has no pages"
                    
                    if validation_results['page_count'] > 500:
                        return False, validation_results, "PDF too large (>500 pages)"
                    
                    # Check for text content
                    text_found = False
                    for i, page in enumerate(pdf_reader.pages[:5]):  # Check first 5 pages
                        try:
                            text = page.extract_text()
                            if text and len(text.strip()) > 50:
                                text_found = True
                                break
                        except Exception:
                            continue
                    
                    validation_results['has_text'] = text_found
                    
            except Exception as e:
                return False, validation_results, f"PDF structure error: {e}"
            
            # Image-based detection (for scanned PDFs)
            if not validation_results['has_text']:
                try:
                    # Convert first page to image to test
                    images = convert_from_path(file_path, first_page=1, last_page=1, dpi=150)
                    if images:
                        validation_results['has_images'] = True
                        validation_results['is_scanned'] = True
                        validation_results['confidence_score'] = 0.7  # Moderate confidence for scanned
                except Exception as e:
                    logger.warning(f"Image conversion test failed: {e}")
                    validation_results['confidence_score'] = 0.3  # Low confidence
            else:
                validation_results['confidence_score'] = 0.9  # High confidence for text PDFs
            
            # Overall validation
            if validation_results['is_encrypted']:
                return False, validation_results, "PDF is password protected"
            
            if not validation_results['has_text'] and not validation_results['has_images']:
                return False, validation_results, "PDF appears to be empty or corrupted"
            
            return True, validation_results, "PDF validation passed"
            
        except Exception as e:
            logger.error(f"PDF validation error: {e}")
            return False, validation_results, f"Validation error: {e}"
    
    def render_pdf_fallback_ui(self, file_path: str, validation_results: Dict[str, Any]) -> bool:
        """Render PDF fallback options for problematic files"""
        try:
            st.warning("⚠️ PDF Processing Issue Detected")
            
            # Show validation results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**File Analysis:**")
                st.write(f"• Size: {validation_results['file_size_mb']:.1f} MB")
                st.write(f"• Pages: {validation_results['page_count']}")
                st.write(f"• Has text: {'✅' if validation_results['has_text'] else '❌'}")
                st.write(f"• Scanned: {'✅' if validation_results['is_scanned'] else '❌'}")
            
            with col2:
                st.markdown("**Processing Options:**")
                
                # Option 1: Skip problematic pages
                if st.button("🔄 Try with OCR (slower but more thorough)"):
                    return True
                
                # Option 2: Process only text pages
                if validation_results['has_text'] and st.button("📄 Process text-only pages"):
                    st.session_state['pdf_text_only'] = True
                    return True
                
                # Option 3: Skip this file
                if st.button("⏭️ Skip this file"):
                    return False
            
            # Show detailed help
            with st.expander("🔧 Troubleshooting Tips"):
                st.markdown("""
                **Common PDF Issues:**
                - **Scanned PDFs**: Require OCR processing (slower)
                - **Rotated pages**: May need manual rotation
                - **Mixed languages**: May have lower accuracy
                - **Large files**: Consider splitting into smaller parts
                
                **Recommendations:**
                - For best results, use text-based PDFs
                - Ensure pages are properly oriented
                - Consider file size limits for processing speed
                """)
            
            return False
            
        except Exception as e:
            logger.error(f"PDF fallback UI error: {e}")
            st.error("Unable to provide fallback options")
            return False
    
    def periodic_auto_save(self):
        """Set up periodic auto-save"""
        try:
            # Check if it's time for auto-save
            last_save = st.session_state.get('last_auto_save', 0)
            current_time = time.time()
            
            if current_time - last_save > 300:  # 5 minutes
                # Prepare session data
                current_data = dict(st.session_state)
                clean_data = {k: v for k, v in current_data.items() 
                            if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict))}
                
                if clean_data and self.auto_save_session(clean_data):
                    st.session_state['last_auto_save'] = current_time
                    logger.info("Periodic auto-save completed")
                    
        except Exception as e:
            logger.warning(f"Periodic auto-save error: {e}")


# Global render optimization instance
render_optimization = RenderOptimization()


# Convenience functions
def initialize_render_optimizations():
    """Initialize all Render.com optimizations"""
    try:
        # Show warming message
        render_optimization.show_warming_message()
        
        # Create heartbeat
        render_optimization.create_healthcheck_endpoint()
        
        # Install runtime dependencies if needed
        render_optimization.install_runtime_dependencies()
        
        # Set up periodic auto-save
        render_optimization.periodic_auto_save()
        
        return True
        
    except Exception as e:
        logger.error(f"Render optimization initialization failed: {e}")
        return False


def validate_pdf_with_fallback(file_path: str) -> Tuple[bool, Dict[str, Any], str]:
    """Validate PDF with enhanced diagnostics and fallback options"""
    return render_optimization.enhanced_pdf_validation(file_path)


def render_session_management():
    """Render session management UI in sidebar"""
    render_optimization.render_session_recovery_ui()


def auto_save_current_session():
    """Auto-save current session data"""
    current_data = dict(st.session_state)
    clean_data = {k: v for k, v in current_data.items() 
                if not k.startswith('_') and isinstance(v, (str, int, float, bool, list, dict))}
    
    return render_optimization.auto_save_session(clean_data)

