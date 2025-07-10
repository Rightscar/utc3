#!/usr/bin/env python3
"""
Progress Tracker Module
Detailed progress tracking with time estimates and user feedback
"""

import time
import math
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import streamlit as st
import logging

class ProgressTracker:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 0
        self.start_time = None
        self.step_times = []
        self.step_names = []
        self.step_descriptions = []
        
        # Initialize session state
        if 'progress_data' not in st.session_state:
            st.session_state.progress_data = {
                'current_step': 0,
                'total_steps': 0,
                'start_time': None,
                'step_times': [],
                'step_names': [],
                'step_descriptions': [],
                'is_active': False,
                'can_cancel': False,
                'is_paused': False
            }
        
        self.logger = logging.getLogger(__name__)
    
    def start_tracking(self, total_steps: int, step_names: List[str], step_descriptions: List[str] = None):
        """Initialize progress tracking"""
        try:
            self.total_steps = total_steps
            self.current_step = 0
            self.start_time = time.time()
            self.step_names = step_names
            self.step_descriptions = step_descriptions or [""] * total_steps
            self.step_times = []
            
            # Update session state
            st.session_state.progress_data.update({
                'current_step': 0,
                'total_steps': total_steps,
                'start_time': self.start_time,
                'step_times': [],
                'step_names': step_names,
                'step_descriptions': self.step_descriptions,
                'is_active': True,
                'can_cancel': True,
                'is_paused': False,
                'last_update': time.time()
            })
            
            self.logger.info(f"Progress tracking started: {total_steps} steps")
            
        except Exception as e:
            self.logger.error(f"Failed to start progress tracking: {str(e)}")
    
    def update_progress(self, step: int, action: str = "", details: str = ""):
        """Update progress with time estimates"""
        try:
            current_time = time.time()
            
            # Record step completion time
            if step > self.current_step and self.current_step > 0:
                step_duration = current_time - st.session_state.progress_data['last_update']
                self.step_times.append(step_duration)
                st.session_state.progress_data['step_times'] = self.step_times
            
            self.current_step = step
            
            # Update session state
            st.session_state.progress_data.update({
                'current_step': step,
                'current_action': action,
                'current_details': details,
                'last_update': current_time,
                'progress_percentage': (step / self.total_steps) * 100 if self.total_steps > 0 else 0
            })
            
            self.logger.info(f"Progress updated: Step {step}/{self.total_steps} - {action}")
            
        except Exception as e:
            self.logger.error(f"Failed to update progress: {str(e)}")
    
    def estimate_remaining_time(self) -> Dict[str, Any]:
        """Calculate estimated time remaining"""
        try:
            if not self.start_time or self.current_step == 0:
                return {
                    'estimated_total': 0,
                    'estimated_remaining': 0,
                    'formatted_remaining': "Calculating...",
                    'completion_time': "Unknown"
                }
            
            current_time = time.time()
            elapsed_time = current_time - self.start_time
            
            # Calculate average time per step
            if self.step_times:
                avg_step_time = sum(self.step_times) / len(self.step_times)
            else:
                avg_step_time = elapsed_time / self.current_step if self.current_step > 0 else 0
            
            # Estimate remaining time
            remaining_steps = self.total_steps - self.current_step
            estimated_remaining = remaining_steps * avg_step_time
            estimated_total = elapsed_time + estimated_remaining
            
            # Calculate completion time
            completion_time = datetime.now() + timedelta(seconds=estimated_remaining)
            
            return {
                'elapsed_time': elapsed_time,
                'estimated_total': estimated_total,
                'estimated_remaining': estimated_remaining,
                'formatted_elapsed': self.format_time(elapsed_time),
                'formatted_remaining': self.format_time(estimated_remaining),
                'formatted_total': self.format_time(estimated_total),
                'completion_time': completion_time.strftime("%H:%M:%S"),
                'avg_step_time': avg_step_time
            }
            
        except Exception as e:
            self.logger.error(f"Time estimation failed: {str(e)}")
            return {
                'estimated_remaining': 0,
                'formatted_remaining': "Unknown",
                'completion_time': "Unknown"
            }
    
    def format_time(self, seconds: float) -> str:
        """Format time in human-readable format"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def show_progress_ui(self, show_details: bool = True, show_cancel: bool = True):
        """Display progress interface in Streamlit"""
        try:
            if not st.session_state.progress_data['is_active']:
                return
            
            progress_data = st.session_state.progress_data
            
            # Main progress bar
            progress_value = progress_data['current_step'] / progress_data['total_steps'] if progress_data['total_steps'] > 0 else 0
            
            # Progress bar with custom styling
            st.markdown("""
            <style>
            .progress-container {
                background-color: #f0f2f6;
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                border: 1px solid #e0e0e0;
            }
            .progress-header {
                font-size: 18px;
                font-weight: bold;
                color: #1f77b4;
                margin-bottom: 10px;
            }
            .progress-details {
                font-size: 14px;
                color: #666;
                margin-top: 10px;
            }
            </style>
            """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown('<div class="progress-container">', unsafe_allow_html=True)
                
                # Header
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f'<div class="progress-header">Processing Step {progress_data["current_step"]}/{progress_data["total_steps"]}</div>', unsafe_allow_html=True)
                with col2:
                    if show_cancel and progress_data['can_cancel']:
                        if st.button("⏹️ Cancel", key="cancel_processing"):
                            self.cancel_processing()
                            return
                
                # Progress bar
                st.progress(progress_value)
                
                # Current action
                current_action = progress_data.get('current_action', '')
                if current_action:
                    st.write(f"**Current Action:** {current_action}")
                
                # Current step name
                if progress_data['current_step'] > 0 and progress_data['current_step'] <= len(progress_data['step_names']):
                    step_name = progress_data['step_names'][progress_data['current_step'] - 1]
                    st.write(f"**Step:** {step_name}")
                
                # Time estimates
                time_estimates = self.estimate_remaining_time()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Elapsed", time_estimates.get('formatted_elapsed', '0s'))
                with col2:
                    st.metric("Remaining", time_estimates.get('formatted_remaining', 'Calculating...'))
                with col3:
                    st.metric("Completion", time_estimates.get('completion_time', 'Unknown'))
                
                # Detailed progress
                if show_details:
                    with st.expander("📊 Detailed Progress", expanded=False):
                        self.show_detailed_progress()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            self.logger.error(f"Progress UI display failed: {str(e)}")
            st.error(f"Progress display error: {str(e)}")
    
    def show_detailed_progress(self):
        """Show detailed progress information"""
        try:
            progress_data = st.session_state.progress_data
            
            # Step breakdown
            st.write("**Step Breakdown:**")
            for i, (name, desc) in enumerate(zip(progress_data['step_names'], progress_data['step_descriptions'])):
                status = "✅" if i < progress_data['current_step'] else "⏳" if i == progress_data['current_step'] else "⏸️"
                st.write(f"{status} **Step {i+1}:** {name}")
                if desc and i == progress_data['current_step']:
                    st.write(f"   _{desc}_")
            
            # Performance metrics
            if progress_data['step_times']:
                st.write("**Performance Metrics:**")
                avg_time = sum(progress_data['step_times']) / len(progress_data['step_times'])
                st.write(f"- Average step time: {self.format_time(avg_time)}")
                st.write(f"- Fastest step: {self.format_time(min(progress_data['step_times']))}")
                st.write(f"- Slowest step: {self.format_time(max(progress_data['step_times']))}")
            
            # System status
            st.write("**System Status:**")
            st.write(f"- Session ID: {st.session_state.get('session_id', 'Unknown')}")
            st.write(f"- Last update: {datetime.fromtimestamp(progress_data['last_update']).strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.logger.error(f"Detailed progress display failed: {str(e)}")
            st.error("Could not display detailed progress")
    
    def pause_processing(self):
        """Pause the current processing"""
        try:
            st.session_state.progress_data['is_paused'] = True
            self.logger.info("Processing paused by user")
            st.info("⏸️ Processing paused. Click Resume to continue.")
            
        except Exception as e:
            self.logger.error(f"Failed to pause processing: {str(e)}")
    
    def resume_processing(self):
        """Resume paused processing"""
        try:
            st.session_state.progress_data['is_paused'] = False
            st.session_state.progress_data['last_update'] = time.time()
            self.logger.info("Processing resumed by user")
            
        except Exception as e:
            self.logger.error(f"Failed to resume processing: {str(e)}")
    
    def cancel_processing(self):
        """Cancel the current processing"""
        try:
            st.session_state.progress_data['is_active'] = False
            st.session_state.progress_data['is_paused'] = False
            self.logger.info("Processing cancelled by user")
            st.warning("⏹️ Processing cancelled by user")
            
            # Auto-save current state
            from .safety_manager import safety_manager
            safety_manager.auto_save_session(force=True)
            
        except Exception as e:
            self.logger.error(f"Failed to cancel processing: {str(e)}")
    
    def complete_processing(self, success: bool = True, message: str = ""):
        """Mark processing as complete"""
        try:
            end_time = time.time()
            total_time = end_time - st.session_state.progress_data['start_time']
            
            st.session_state.progress_data.update({
                'is_active': False,
                'is_paused': False,
                'completed': True,
                'success': success,
                'completion_message': message,
                'total_time': total_time,
                'end_time': end_time
            })
            
            if success:
                st.success(f"✅ Processing completed successfully! Total time: {self.format_time(total_time)}")
                if message:
                    st.info(message)
            else:
                st.error(f"❌ Processing failed. Time elapsed: {self.format_time(total_time)}")
                if message:
                    st.error(message)
            
            self.logger.info(f"Processing completed: Success={success}, Time={self.format_time(total_time)}")
            
        except Exception as e:
            self.logger.error(f"Failed to complete processing: {str(e)}")
    
    def get_progress_stats(self) -> Dict[str, Any]:
        """Get current progress statistics"""
        try:
            progress_data = st.session_state.progress_data
            time_estimates = self.estimate_remaining_time()
            
            return {
                'current_step': progress_data['current_step'],
                'total_steps': progress_data['total_steps'],
                'progress_percentage': progress_data.get('progress_percentage', 0),
                'is_active': progress_data['is_active'],
                'is_paused': progress_data['is_paused'],
                'elapsed_time': time_estimates.get('elapsed_time', 0),
                'estimated_remaining': time_estimates.get('estimated_remaining', 0),
                'completion_time': time_estimates.get('completion_time', 'Unknown'),
                'current_action': progress_data.get('current_action', ''),
                'avg_step_time': time_estimates.get('avg_step_time', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get progress stats: {str(e)}")
            return {'error': str(e)}
    
    def show_mini_progress(self):
        """Show minimal progress indicator"""
        try:
            if not st.session_state.progress_data['is_active']:
                return
            
            progress_data = st.session_state.progress_data
            progress_value = progress_data['current_step'] / progress_data['total_steps'] if progress_data['total_steps'] > 0 else 0
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.progress(progress_value)
            
            with col2:
                st.write(f"{progress_data['current_step']}/{progress_data['total_steps']}")
            
            with col3:
                time_estimates = self.estimate_remaining_time()
                st.write(f"⏱️ {time_estimates.get('formatted_remaining', '...')}")
            
        except Exception as e:
            self.logger.error(f"Mini progress display failed: {str(e)}")
    
    def is_processing(self) -> bool:
        """Check if processing is currently active"""
        return st.session_state.progress_data.get('is_active', False)
    
    def is_paused(self) -> bool:
        """Check if processing is paused"""
        return st.session_state.progress_data.get('is_paused', False)
    
    def reset_progress(self):
        """Reset progress tracking"""
        try:
            st.session_state.progress_data = {
                'current_step': 0,
                'total_steps': 0,
                'start_time': None,
                'step_times': [],
                'step_names': [],
                'step_descriptions': [],
                'is_active': False,
                'can_cancel': False,
                'is_paused': False
            }
            
            self.current_step = 0
            self.total_steps = 0
            self.start_time = None
            self.step_times = []
            self.step_names = []
            self.step_descriptions = []
            
            self.logger.info("Progress tracking reset")
            
        except Exception as e:
            self.logger.error(f"Failed to reset progress: {str(e)}")
    
    def create_progress_report(self) -> Dict[str, Any]:
        """Create comprehensive progress report"""
        try:
            progress_data = st.session_state.progress_data
            
            report = {
                'session_id': st.session_state.get('session_id', 'Unknown'),
                'start_time': datetime.fromtimestamp(progress_data['start_time']).isoformat() if progress_data['start_time'] else None,
                'end_time': datetime.fromtimestamp(progress_data.get('end_time', time.time())).isoformat(),
                'total_steps': progress_data['total_steps'],
                'completed_steps': progress_data['current_step'],
                'success': progress_data.get('success', False),
                'total_time': progress_data.get('total_time', 0),
                'step_times': progress_data['step_times'],
                'step_names': progress_data['step_names'],
                'avg_step_time': sum(progress_data['step_times']) / len(progress_data['step_times']) if progress_data['step_times'] else 0,
                'completion_message': progress_data.get('completion_message', '')
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to create progress report: {str(e)}")
            return {'error': str(e)}

# Global instance
progress_tracker = ProgressTracker()

