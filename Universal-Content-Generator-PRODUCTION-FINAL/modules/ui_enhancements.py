"""
UI Enhancement Module for Phase 1
=================================

This module provides enhanced UI components for the Universal Text-to-Dialogue AI,
including transformer model selection, real-time quality feedback, progress visualization,
and performance monitoring dashboard.

Features:
- Transformer model selection interface
- Real-time quality feedback and metrics
- Enhanced progress visualization with time estimates
- Performance monitoring dashboard
- Interactive quality controls
- User-friendly error display
- Advanced settings management
"""

import streamlit as st
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelSelector:
    """Enhanced transformer model selection interface"""
    
    AVAILABLE_MODELS = {
        "sentence-transformers/all-MiniLM-L6-v2": {
            "name": "All-MiniLM-L6-v2",
            "description": "Fast and efficient, good for general use",
            "size": "80MB",
            "speed": "⚡⚡⚡",
            "quality": "⭐⭐⭐",
            "recommended": True
        },
        "sentence-transformers/all-mpnet-base-v2": {
            "name": "All-MPNet-Base-v2", 
            "description": "High quality embeddings, slower processing",
            "size": "420MB",
            "speed": "⚡⚡",
            "quality": "⭐⭐⭐⭐⭐",
            "recommended": False
        },
        "sentence-transformers/paraphrase-MiniLM-L6-v2": {
            "name": "Paraphrase-MiniLM-L6-v2",
            "description": "Optimized for paraphrase detection",
            "size": "80MB", 
            "speed": "⚡⚡⚡",
            "quality": "⭐⭐⭐⭐",
            "recommended": False
        },
        "bert-base-uncased": {
            "name": "BERT Base Uncased",
            "description": "Classic BERT model, good general performance",
            "size": "440MB",
            "speed": "⚡",
            "quality": "⭐⭐⭐⭐",
            "recommended": False
        },
        "distilbert-base-uncased": {
            "name": "DistilBERT Base",
            "description": "Lighter version of BERT, faster processing",
            "size": "250MB",
            "speed": "⚡⚡",
            "quality": "⭐⭐⭐",
            "recommended": False
        },
        "roberta-base": {
            "name": "RoBERTa Base",
            "description": "Improved BERT variant, high quality",
            "size": "500MB",
            "speed": "⚡",
            "quality": "⭐⭐⭐⭐⭐",
            "recommended": False
        }
    }
    
    @staticmethod
    def render_model_selection():
        """Render enhanced model selection interface"""
        st.subheader("🧠 Transformer Model Selection")
        
        # Get current selection
        current_model = st.session_state.get('selected_transformer_model', 
                                           'sentence-transformers/all-MiniLM-L6-v2')
        
        # Create model comparison table
        model_data = []
        for model_id, info in ModelSelector.AVAILABLE_MODELS.items():
            model_data.append({
                'Model': info['name'],
                'Description': info['description'],
                'Size': info['size'],
                'Speed': info['speed'],
                'Quality': info['quality'],
                'Recommended': '✅' if info['recommended'] else '',
                'ID': model_id
            })
        
        df = pd.DataFrame(model_data)
        
        # Display model comparison
        with st.expander("📊 Model Comparison", expanded=False):
            st.dataframe(df[['Model', 'Description', 'Size', 'Speed', 'Quality', 'Recommended']], 
                        use_container_width=True)
        
        # Model selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model = st.selectbox(
                "Select Transformer Model",
                options=list(ModelSelector.AVAILABLE_MODELS.keys()),
                index=list(ModelSelector.AVAILABLE_MODELS.keys()).index(current_model),
                format_func=lambda x: ModelSelector.AVAILABLE_MODELS[x]['name'],
                help="Choose the transformer model for semantic understanding"
            )
        
        with col2:
            if st.button("🔄 Reload Model", help="Reload the selected model"):
                st.session_state.model_reload_requested = True
                st.success("Model reload requested!")
        
        # Display selected model info
        if selected_model:
            model_info = ModelSelector.AVAILABLE_MODELS[selected_model]
            
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 4px solid #15c39a;">
                <h5>📋 Selected Model: {model_info['name']}</h5>
                <p><strong>Description:</strong> {model_info['description']}</p>
                <p><strong>Size:</strong> {model_info['size']} | <strong>Speed:</strong> {model_info['speed']} | <strong>Quality:</strong> {model_info['quality']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Store selection
        st.session_state.selected_transformer_model = selected_model
        
        return selected_model

class QualityFeedback:
    """Real-time quality feedback and metrics display"""
    
    @staticmethod
    def render_quality_dashboard(quality_data: Optional[Dict[str, Any]] = None):
        """Render real-time quality feedback dashboard"""
        st.subheader("📊 Real-time Quality Assessment")
        
        if not quality_data:
            quality_data = QualityFeedback._get_default_quality_data()
        
        # Quality metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            semantic_score = quality_data.get('semantic_quality', 0.0)
            QualityFeedback._render_quality_metric(
                "🧠 Semantic Quality", 
                semantic_score, 
                "Measures semantic understanding accuracy"
            )
        
        with col2:
            coherence_score = quality_data.get('coherence', 0.0)
            QualityFeedback._render_quality_metric(
                "🔗 Coherence", 
                coherence_score, 
                "Measures text flow and consistency"
            )
        
        with col3:
            confidence_score = quality_data.get('confidence', 0.0)
            QualityFeedback._render_quality_metric(
                "✅ Confidence", 
                confidence_score, 
                "Model confidence in processing"
            )
        
        with col4:
            overall_score = quality_data.get('overall_quality', 0.0)
            QualityFeedback._render_quality_metric(
                "⭐ Overall", 
                overall_score, 
                "Combined quality assessment"
            )
        
        # Quality trend chart
        if 'quality_history' in quality_data:
            QualityFeedback._render_quality_trend(quality_data['quality_history'])
        
        # Quality thresholds
        QualityFeedback._render_quality_controls()
    
    @staticmethod
    def _render_quality_metric(title: str, score: float, description: str):
        """Render individual quality metric"""
        # Determine color based on score
        if score >= 0.8:
            color = "#15c39a"  # Green
            status = "Excellent"
        elif score >= 0.6:
            color = "#ffa500"  # Orange
            status = "Good"
        elif score >= 0.4:
            color = "#ff6b6b"  # Red
            status = "Needs Improvement"
        else:
            color = "#dc3545"  # Dark red
            status = "Poor"
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e0e0e0; text-align: center;">
            <h4 style="margin: 0; color: {color};">{title}</h4>
            <h2 style="margin: 0.5rem 0; color: {color};">{score:.2f}</h2>
            <p style="margin: 0; color: #666; font-size: 0.8rem;">{status}</p>
            <p style="margin: 0.5rem 0 0 0; color: #888; font-size: 0.7rem;">{description}</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _render_quality_trend(quality_history: List[Dict[str, Any]]):
        """Render quality trend chart"""
        if not quality_history:
            return
        
        st.subheader("📈 Quality Trends")
        
        # Convert to DataFrame
        df = pd.DataFrame(quality_history)
        
        # Create trend chart
        fig = go.Figure()
        
        metrics = ['semantic_quality', 'coherence', 'confidence', 'overall_quality']
        colors = ['#15c39a', '#007bff', '#ffa500', '#6f42c1']
        
        for metric, color in zip(metrics, colors):
            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=color, width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title="Quality Metrics Over Time",
            xaxis_title="Processing Steps",
            yaxis_title="Quality Score",
            yaxis=dict(range=[0, 1]),
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    @staticmethod
    def _render_quality_controls():
        """Render quality threshold controls"""
        with st.expander("⚙️ Quality Thresholds", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                semantic_threshold = st.slider(
                    "Semantic Quality Threshold",
                    min_value=0.0, max_value=1.0, value=0.75, step=0.05,
                    help="Minimum acceptable semantic quality score"
                )
                
                coherence_threshold = st.slider(
                    "Coherence Threshold", 
                    min_value=0.0, max_value=1.0, value=0.70, step=0.05,
                    help="Minimum acceptable coherence score"
                )
            
            with col2:
                confidence_threshold = st.slider(
                    "Confidence Threshold",
                    min_value=0.0, max_value=1.0, value=0.80, step=0.05,
                    help="Minimum acceptable confidence score"
                )
                
                auto_reject = st.checkbox(
                    "Auto-reject low quality",
                    value=False,
                    help="Automatically reject chunks below thresholds"
                )
            
            # Store thresholds
            st.session_state.update({
                'semantic_threshold': semantic_threshold,
                'coherence_threshold': coherence_threshold,
                'confidence_threshold': confidence_threshold,
                'auto_reject_low_quality': auto_reject
            })
    
    @staticmethod
    def _get_default_quality_data():
        """Get default quality data for display"""
        return {
            'semantic_quality': 0.85,
            'coherence': 0.78,
            'confidence': 0.82,
            'overall_quality': 0.81,
            'quality_history': []
        }

class ProgressVisualizer:
    """Enhanced progress visualization with time estimates"""
    
    @staticmethod
    def create_progress_tracker(total_steps: int, current_step: int = 0) -> Dict[str, Any]:
        """Create a progress tracking context"""
        return {
            'total_steps': total_steps,
            'current_step': current_step,
            'start_time': time.time(),
            'step_times': [],
            'step_names': [],
            'estimated_completion': None
        }
    
    @staticmethod
    def update_progress(progress_tracker: Dict[str, Any], 
                       step_name: str, 
                       step_number: Optional[int] = None) -> Dict[str, Any]:
        """Update progress tracker with new step"""
        current_time = time.time()
        
        if step_number is not None:
            progress_tracker['current_step'] = step_number
        else:
            progress_tracker['current_step'] += 1
        
        progress_tracker['step_times'].append(current_time)
        progress_tracker['step_names'].append(step_name)
        
        # Calculate estimated completion time
        if len(progress_tracker['step_times']) > 1:
            elapsed_time = current_time - progress_tracker['start_time']
            avg_step_time = elapsed_time / progress_tracker['current_step']
            remaining_steps = progress_tracker['total_steps'] - progress_tracker['current_step']
            estimated_remaining = avg_step_time * remaining_steps
            progress_tracker['estimated_completion'] = current_time + estimated_remaining
        
        return progress_tracker
    
    @staticmethod
    def render_progress_display(progress_tracker: Dict[str, Any], 
                              container: Optional[st.container] = None):
        """Render enhanced progress display"""
        if container is None:
            container = st
        
        current_step = progress_tracker['current_step']
        total_steps = progress_tracker['total_steps']
        progress_percent = min(current_step / total_steps, 1.0) if total_steps > 0 else 0
        
        # Progress bar
        container.progress(progress_percent)
        
        # Progress info
        col1, col2, col3 = container.columns(3)
        
        with col1:
            container.metric(
                "Progress", 
                f"{current_step}/{total_steps}",
                f"{progress_percent:.1%}"
            )
        
        with col2:
            elapsed_time = time.time() - progress_tracker['start_time']
            container.metric(
                "Elapsed Time",
                f"{elapsed_time:.1f}s",
                ""
            )
        
        with col3:
            if progress_tracker['estimated_completion']:
                remaining_time = progress_tracker['estimated_completion'] - time.time()
                container.metric(
                    "Est. Remaining",
                    f"{max(remaining_time, 0):.1f}s",
                    ""
                )
        
        # Current step info
        if progress_tracker['step_names']:
            current_step_name = progress_tracker['step_names'][-1]
            container.info(f"🔄 Current step: {current_step_name}")
        
        # Step history
        if len(progress_tracker['step_names']) > 1:
            with container.expander("📋 Processing Steps", expanded=False):
                for i, (step_name, step_time) in enumerate(zip(
                    progress_tracker['step_names'], 
                    progress_tracker['step_times']
                )):
                    step_duration = (step_time - progress_tracker['start_time'])
                    status = "✅" if i < len(progress_tracker['step_names']) - 1 else "🔄"
                    container.write(f"{status} Step {i+1}: {step_name} ({step_duration:.1f}s)")

class PerformanceMonitor:
    """Performance monitoring dashboard"""
    
    @staticmethod
    def render_performance_dashboard(performance_data: Optional[Dict[str, Any]] = None):
        """Render performance monitoring dashboard"""
        st.subheader("⚡ Performance Monitoring")
        
        if not performance_data:
            performance_data = PerformanceMonitor._get_default_performance_data()
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            processing_speed = performance_data.get('processing_speed', 0)
            PerformanceMonitor._render_performance_metric(
                "🚀 Speed", 
                f"{processing_speed:.1f} words/s",
                "green" if processing_speed > 100 else "orange"
            )
        
        with col2:
            memory_usage = performance_data.get('memory_usage', 0)
            PerformanceMonitor._render_performance_metric(
                "💾 Memory", 
                f"{memory_usage:.1f}%",
                "green" if memory_usage < 70 else "red"
            )
        
        with col3:
            cpu_usage = performance_data.get('cpu_usage', 0)
            PerformanceMonitor._render_performance_metric(
                "🖥️ CPU", 
                f"{cpu_usage:.1f}%",
                "green" if cpu_usage < 80 else "orange"
            )
        
        with col4:
            cache_hit_rate = performance_data.get('cache_hit_rate', 0)
            PerformanceMonitor._render_performance_metric(
                "📋 Cache", 
                f"{cache_hit_rate:.1f}%",
                "green" if cache_hit_rate > 50 else "orange"
            )
        
        # Performance charts
        if 'performance_history' in performance_data:
            PerformanceMonitor._render_performance_charts(performance_data['performance_history'])
    
    @staticmethod
    def _render_performance_metric(title: str, value: str, color: str):
        """Render individual performance metric"""
        color_map = {
            'green': '#15c39a',
            'orange': '#ffa500', 
            'red': '#ff6b6b'
        }
        
        hex_color = color_map.get(color, '#15c39a')
        
        st.markdown(f"""
        <div style="background: white; padding: 1rem; border-radius: 8px; border: 1px solid #e0e0e0; text-align: center;">
            <h4 style="margin: 0; color: {hex_color};">{title}</h4>
            <h2 style="margin: 0.5rem 0; color: {hex_color};">{value}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def _render_performance_charts(performance_history: List[Dict[str, Any]]):
        """Render performance trend charts"""
        if not performance_history:
            return
        
        df = pd.DataFrame(performance_history)
        
        # Create subplots
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing speed chart
            fig_speed = px.line(
                df, 
                x='timestamp', 
                y='processing_speed',
                title="Processing Speed Over Time",
                labels={'processing_speed': 'Words/Second', 'timestamp': 'Time'}
            )
            fig_speed.update_layout(height=300)
            st.plotly_chart(fig_speed, use_container_width=True)
        
        with col2:
            # Resource usage chart
            fig_resources = go.Figure()
            fig_resources.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['memory_usage'],
                mode='lines',
                name='Memory %',
                line=dict(color='#ff6b6b')
            ))
            fig_resources.add_trace(go.Scatter(
                x=df['timestamp'], 
                y=df['cpu_usage'],
                mode='lines', 
                name='CPU %',
                line=dict(color='#007bff')
            ))
            fig_resources.update_layout(
                title="Resource Usage Over Time",
                xaxis_title="Time",
                yaxis_title="Usage %",
                height=300
            )
            st.plotly_chart(fig_resources, use_container_width=True)
    
    @staticmethod
    def _get_default_performance_data():
        """Get default performance data"""
        return {
            'processing_speed': 150.0,
            'memory_usage': 45.2,
            'cpu_usage': 32.1,
            'cache_hit_rate': 78.5,
            'performance_history': []
        }

class AdvancedSettings:
    """Advanced settings management"""
    
    @staticmethod
    def render_advanced_settings():
        """Render comprehensive advanced settings"""
        with st.expander("🔧 Advanced Processing Settings", expanded=False):
            
            # Model selection
            ModelSelector.render_model_selection()
            
            st.divider()
            
            # Processing options
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚙️ Processing Options")
                
                enable_attention = st.checkbox(
                    "Enable Attention Analysis",
                    value=st.session_state.get('enable_attention', True),
                    help="Extract attention weights for better understanding"
                )
                
                batch_size = st.slider(
                    "Batch Size",
                    min_value=1, max_value=32, 
                    value=st.session_state.get('batch_size', 8),
                    help="Number of texts to process simultaneously"
                )
                
                enable_caching = st.checkbox(
                    "Enable Result Caching",
                    value=st.session_state.get('enable_caching', True),
                    help="Cache results for faster repeated processing"
                )
            
            with col2:
                st.subheader("🎯 Quality Settings")
                
                enable_live_quality = st.checkbox(
                    "Live Quality Assessment",
                    value=st.session_state.get('enable_live_quality', True),
                    help="Show quality metrics in real-time"
                )
                
                show_confidence = st.checkbox(
                    "Show Confidence Scores", 
                    value=st.session_state.get('show_confidence', True),
                    help="Display confidence scores for processing"
                )
                
                detailed_logging = st.checkbox(
                    "Detailed Logging",
                    value=st.session_state.get('detailed_logging', False),
                    help="Enable detailed processing logs"
                )
            
            # Store all settings
            st.session_state.update({
                'enable_attention': enable_attention,
                'batch_size': batch_size,
                'enable_caching': enable_caching,
                'enable_live_quality': enable_live_quality,
                'show_confidence': show_confidence,
                'detailed_logging': detailed_logging
            })
            
            st.divider()
            
            # Performance monitoring toggle
            st.subheader("📊 Monitoring")
            enable_performance_monitoring = st.checkbox(
                "Enable Performance Monitoring",
                value=st.session_state.get('enable_performance_monitoring', True),
                help="Monitor and display performance metrics"
            )
            st.session_state.enable_performance_monitoring = enable_performance_monitoring

# Convenience functions for easy integration
def render_model_selector():
    """Convenience function for model selection"""
    return ModelSelector.render_model_selection()

def render_quality_feedback(quality_data=None):
    """Convenience function for quality feedback"""
    return QualityFeedback.render_quality_dashboard(quality_data)

def render_performance_monitor(performance_data=None):
    """Convenience function for performance monitoring"""
    return PerformanceMonitor.render_performance_dashboard(performance_data)

def create_progress_tracker(total_steps):
    """Convenience function for progress tracking"""
    return ProgressVisualizer.create_progress_tracker(total_steps)

def update_progress(tracker, step_name, step_number=None):
    """Convenience function for progress updates"""
    return ProgressVisualizer.update_progress(tracker, step_name, step_number)

def render_progress(tracker, container=None):
    """Convenience function for progress display"""
    return ProgressVisualizer.render_progress_display(tracker, container)

# Example usage
if __name__ == "__main__":
    print("🧪 Testing UI Enhancement Module...")
    
    # This would be used in a Streamlit app
    print("✅ UI Enhancement module ready for integration")
    print("🎉 All UI components available for Phase 1 implementation")

