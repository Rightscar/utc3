"""
Enhanced UI Polish Module
========================

Provides advanced UI/UX enhancements for the Enhanced Universal AI Training Data Creator.
Includes collapsible sections, breadcrumb navigation, progress indicators, and improved user experience.

Features:
- Collapsible sections for long content
- Breadcrumb navigation and stepper UI
- Progress indicators and status tracking
- Dark mode toggle integration
- Responsive design helpers
- User guidance and tooltips
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class EnhancedUIPolish:
    """Enhanced UI polish system with advanced UX components"""
    
    def __init__(self):
        self.step_names = [
            "Upload & Extract",
            "Content Analysis", 
            "Enhancement",
            "Review & Validate",
            "Export & Share"
        ]
        self.current_step = 1
        
    def render_breadcrumb_navigation(self, current_step: int = 1):
        """Render breadcrumb navigation with step indicators"""
        st.markdown("### 🗺️ **Workflow Progress**")
        
        # Create step indicator
        cols = st.columns(len(self.step_names))
        
        for i, (col, step_name) in enumerate(zip(cols, self.step_names), 1):
            with col:
                if i < current_step:
                    # Completed step
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background-color: #28a745; 
                                color: white; border-radius: 5px; margin: 2px;">
                        <strong>✅ Step {i}</strong><br>
                        <small>{step_name}</small>
                    </div>
                    """, unsafe_allow_html=True)
                elif i == current_step:
                    # Current step
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background-color: #007bff; 
                                color: white; border-radius: 5px; margin: 2px; 
                                border: 2px solid #0056b3;">
                        <strong>🔄 Step {i}</strong><br>
                        <small>{step_name}</small>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Future step
                    st.markdown(f"""
                    <div style="text-align: center; padding: 10px; background-color: #6c757d; 
                                color: white; border-radius: 5px; margin: 2px;">
                        <strong>⏳ Step {i}</strong><br>
                        <small>{step_name}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    def render_collapsible_section(self, title: str, content_func, 
                                 default_expanded: bool = False, 
                                 help_text: str = None):
        """Render collapsible section with optional help text"""
        with st.expander(title, expanded=default_expanded):
            if help_text:
                st.info(f"💡 **Tip:** {help_text}")
            content_func()
    
    def render_advanced_analysis_section(self, content_stats: Dict[str, Any]):
        """Render advanced analysis in collapsible sections"""
        
        def basic_stats():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Characters", f"{content_stats.get('total_chars', 0):,}")
            
            with col2:
                st.metric("Total Words", f"{content_stats.get('total_words', 0):,}")
            
            with col3:
                st.metric("Total Lines", f"{content_stats.get('total_lines', 0):,}")
            
            with col4:
                st.metric("Avg Words/Line", f"{content_stats.get('avg_words_per_line', 0):.1f}")
        
        def detailed_analysis():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Content Quality Metrics:**")
                complexity = content_stats.get('complexity_score', 0)
                readability = content_stats.get('readability_score', 0)
                dialogue_ratio = content_stats.get('dialogue_ratio', 0)
                
                st.progress(complexity, text=f"Complexity Score: {complexity:.2f}")
                st.progress(readability, text=f"Readability Score: {readability:.2f}")
                st.progress(dialogue_ratio, text=f"Dialogue Ratio: {dialogue_ratio:.2f}")
            
            with col2:
                st.markdown("**Content Distribution:**")
                
                # Create a simple chart representation
                if content_stats.get('total_lines', 0) > 0:
                    chart_data = {
                        'Short Lines (<5 words)': content_stats.get('short_lines', 0),
                        'Medium Lines (5-15 words)': content_stats.get('medium_lines', 0),
                        'Long Lines (>15 words)': content_stats.get('long_lines', 0)
                    }
                    
                    for category, count in chart_data.items():
                        percentage = (count / content_stats['total_lines']) * 100
                        st.write(f"{category}: {count} ({percentage:.1f}%)")
        
        # Render sections
        self.render_collapsible_section(
            "📊 Basic Statistics", 
            basic_stats, 
            default_expanded=True,
            help_text="Overview of your content's basic metrics"
        )
        
        self.render_collapsible_section(
            "🔍 Advanced Analysis", 
            detailed_analysis,
            default_expanded=False,
            help_text="Detailed quality and distribution analysis"
        )
    
    def render_enhancement_progress(self, progress_data: Dict[str, Any]):
        """Render enhancement progress with visual indicators"""
        
        def progress_overview():
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_items = progress_data.get('total_items', 0)
                st.metric("Total Items", total_items)
            
            with col2:
                processed_items = progress_data.get('processed_items', 0)
                st.metric("Processed", processed_items)
            
            with col3:
                if total_items > 0:
                    completion_rate = (processed_items / total_items) * 100
                    st.metric("Completion", f"{completion_rate:.1f}%")
                else:
                    st.metric("Completion", "0%")
            
            # Progress bar
            if total_items > 0:
                progress = processed_items / total_items
                st.progress(progress, text=f"Processing: {processed_items}/{total_items}")
        
        def detailed_progress():
            # Enhancement statistics
            enhancement_stats = progress_data.get('enhancement_stats', {})
            
            if enhancement_stats:
                st.markdown("**Enhancement Statistics:**")
                
                for tone, stats in enhancement_stats.items():
                    with st.container():
                        st.markdown(f"**{tone.replace('_', ' ').title()}:**")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"Items: {stats.get('count', 0)}")
                        
                        with col2:
                            avg_time = stats.get('avg_time', 0)
                            st.write(f"Avg Time: {avg_time:.2f}s")
                        
                        with col3:
                            total_cost = stats.get('total_cost', 0)
                            st.write(f"Cost: ${total_cost:.4f}")
        
        self.render_collapsible_section(
            "⏳ Progress Overview",
            progress_overview,
            default_expanded=True,
            help_text="Track the progress of your content enhancement"
        )
        
        self.render_collapsible_section(
            "📈 Detailed Progress",
            detailed_progress,
            default_expanded=False,
            help_text="Detailed statistics about the enhancement process"
        )
    
    def render_quality_dashboard(self, quality_data: Dict[str, Any]):
        """Render quality assessment dashboard"""
        
        def quality_overview():
            overall_score = quality_data.get('overall_score', 0)
            passed_threshold = quality_data.get('passed_threshold', False)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score_color = "green" if overall_score >= 0.7 else "orange" if overall_score >= 0.5 else "red"
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: {score_color}; 
                            color: white; border-radius: 10px;">
                    <h2>{overall_score:.2f}</h2>
                    <p>Overall Quality Score</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                status_icon = "✅" if passed_threshold else "❌"
                status_text = "PASSED" if passed_threshold else "NEEDS REVIEW"
                status_color = "green" if passed_threshold else "red"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: {status_color}; 
                            color: white; border-radius: 10px;">
                    <h2>{status_icon}</h2>
                    <p>{status_text}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                flags = quality_data.get('flags', [])
                flag_count = len(flags)
                flag_color = "green" if flag_count == 0 else "orange" if flag_count <= 2 else "red"
                
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: {flag_color}; 
                            color: white; border-radius: 10px;">
                    <h2>{flag_count}</h2>
                    <p>Quality Flags</p>
                </div>
                """, unsafe_allow_html=True)
        
        def quality_metrics():
            metrics = quality_data.get('metrics', {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Semantic Analysis:**")
                
                semantic_similarity = metrics.get('semantic_similarity', 0)
                st.progress(semantic_similarity, text=f"Semantic Similarity: {semantic_similarity:.3f}")
                
                hallucination_score = metrics.get('hallucination_score', 0)
                hallucination_display = 1 - hallucination_score  # Invert for display
                st.progress(hallucination_display, text=f"Content Reliability: {hallucination_display:.3f}")
            
            with col2:
                st.markdown("**Content Quality:**")
                
                readability = metrics.get('readability_score', 0)
                st.progress(readability, text=f"Readability: {readability:.3f}")
                
                coherence = metrics.get('coherence_score', 0)
                st.progress(coherence, text=f"Coherence: {coherence:.3f}")
        
        def quality_flags():
            flags = quality_data.get('flags', [])
            recommendations = quality_data.get('recommendations', [])
            
            if flags:
                st.markdown("**Quality Flags:**")
                for flag in flags:
                    st.warning(f"⚠️ {flag.replace('_', ' ').title()}")
            else:
                st.success("✅ No quality issues detected!")
            
            if recommendations:
                st.markdown("**Recommendations:**")
                for rec in recommendations:
                    st.info(f"💡 {rec}")
        
        self.render_collapsible_section(
            "🎯 Quality Overview",
            quality_overview,
            default_expanded=True,
            help_text="Overall quality assessment of your enhanced content"
        )
        
        self.render_collapsible_section(
            "📊 Quality Metrics",
            quality_metrics,
            default_expanded=True,
            help_text="Detailed quality metrics and scores"
        )
        
        self.render_collapsible_section(
            "🚩 Flags & Recommendations",
            quality_flags,
            default_expanded=False,
            help_text="Quality issues and improvement suggestions"
        )
    
    def render_step_completion_status(self, step_statuses: Dict[str, bool]):
        """Render step completion status with enable/disable logic"""
        
        st.markdown("### 📋 **Step Completion Status**")
        
        for i, step_name in enumerate(self.step_names, 1):
            step_key = f"step_{i}"
            is_completed = step_statuses.get(step_key, False)
            is_current = i == self.current_step
            is_accessible = i == 1 or step_statuses.get(f"step_{i-1}", False)
            
            # Create status indicator
            if is_completed:
                status_icon = "✅"
                status_color = "green"
                status_text = "Completed"
            elif is_current and is_accessible:
                status_icon = "🔄"
                status_color = "blue"
                status_text = "In Progress"
            elif is_accessible:
                status_icon = "⏳"
                status_color = "orange"
                status_text = "Ready"
            else:
                status_icon = "🔒"
                status_color = "gray"
                status_text = "Locked"
            
            col1, col2, col3 = st.columns([1, 3, 1])
            
            with col1:
                st.markdown(f"**Step {i}**")
            
            with col2:
                st.markdown(f"{status_icon} {step_name}")
            
            with col3:
                st.markdown(f"<span style='color: {status_color}'>{status_text}</span>", 
                           unsafe_allow_html=True)
    
    def render_reprocess_controls(self, step_statuses: Dict[str, bool]):
        """Render reprocess controls with clear tooltips"""
        
        st.markdown("### 🔄 **Reprocess Controls**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if step_statuses.get('step_1', False):
                if st.button("🔄 Re-extract Content", 
                           help="Re-extract content from uploaded files. This will reset all subsequent steps."):
                    # Reset subsequent steps
                    for i in range(2, 6):
                        if f'step_{i}' in st.session_state:
                            st.session_state[f'step_{i}'] = False
                    st.rerun()
            else:
                st.button("🔄 Re-extract Content", disabled=True, 
                         help="Please complete content extraction first.")
        
        with col2:
            if step_statuses.get('step_3', False):
                if st.button("🔄 Re-enhance Content", 
                           help="Re-enhance content with AI. This will reset review and export steps."):
                    # Reset subsequent steps
                    for i in range(4, 6):
                        if f'step_{i}' in st.session_state:
                            st.session_state[f'step_{i}'] = False
                    st.rerun()
            else:
                st.button("🔄 Re-enhance Content", disabled=True, 
                         help="Please complete content enhancement first.")
        
        with col3:
            if step_statuses.get('step_4', False):
                if st.button("🔄 Re-validate Quality", 
                           help="Re-run quality validation on enhanced content."):
                    # Reset export step
                    if 'step_5' in st.session_state:
                        st.session_state['step_5'] = False
                    st.rerun()
            else:
                st.button("🔄 Re-validate Quality", disabled=True, 
                         help="Please complete quality validation first.")
    
    def render_dark_mode_toggle(self):
        """Render dark mode toggle (integrates with enhanced theming)"""
        
        # Check if dark mode is enabled
        current_theme = st.session_state.get('selected_theme', 'default')
        is_dark_mode = current_theme == 'dark'
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🌙" if not is_dark_mode else "☀️", 
                        help="Toggle dark mode"):
                if is_dark_mode:
                    st.session_state['selected_theme'] = 'default'
                else:
                    st.session_state['selected_theme'] = 'dark'
                st.rerun()
    
    def render_help_tooltips(self, section: str):
        """Render contextual help tooltips for different sections"""
        
        help_content = {
            'upload': {
                'title': "📤 Upload & Extract Help",
                'content': """
                **Supported Formats:** PDF, TXT, DOCX, MD
                
                **Tips:**
                - Ensure files contain readable text (not scanned images)
                - Larger files may take longer to process
                - Multiple files can be processed in sequence
                
                **Troubleshooting:**
                - If extraction fails, try a different file format
                - Check that the file isn't corrupted
                - Ensure the file contains actual text content
                """
            },
            'analysis': {
                'title': "🔍 Content Analysis Help",
                'content': """
                **Content Types:**
                - **Q&A:** Question and answer format
                - **Dialogue:** Conversational exchanges
                - **Monologue:** Continuous narrative text
                
                **Quality Metrics:**
                - **Complexity:** Vocabulary diversity
                - **Readability:** Sentence and word length
                - **Dialogue Ratio:** Proportion of conversational content
                """
            },
            'enhancement': {
                'title': "✨ Enhancement Help",
                'content': """
                **Spiritual Tones Available:**
                - Zen Buddhism: Mindful, present-moment awareness
                - Advaita Vedanta: Non-dual consciousness exploration
                - Christian Mysticism: Contemplative spiritual practice
                - Sufi Mysticism: Heart-centered divine love
                - Mindfulness Meditation: Present-moment awareness
                - Universal Wisdom: Cross-traditional insights
                
                **Enhancement Process:**
                - AI analyzes your content's meaning
                - Applies selected spiritual tone and style
                - Maintains core message while enhancing expression
                """
            },
            'review': {
                'title': "📋 Review & Validation Help",
                'content': """
                **Quality Assessment:**
                - **Semantic Similarity:** How well enhanced content matches original meaning
                - **Hallucination Score:** Detection of unsupported claims
                - **Content Quality:** Readability, coherence, completeness
                
                **Manual Review:**
                - Edit enhanced content directly
                - Approve or reject individual items
                - Add custom notes and feedback
                """
            },
            'export': {
                'title': "📦 Export & Share Help",
                'content': """
                **Export Formats:**
                - **JSON/JSONL:** For AI training pipelines
                - **CSV/XLSX:** For data analysis
                - **TXT:** For simple text processing
                - **ZIP:** Comprehensive package with documentation
                
                **Sharing Options:**
                - **Hugging Face:** Direct upload to ML platform
                - **Local Download:** Save to your computer
                - **Documentation:** Includes metadata and quality reports
                """
            }
        }
        
        if section in help_content:
            help_info = help_content[section]
            
            with st.expander(f"❓ {help_info['title']}", expanded=False):
                st.markdown(help_info['content'])
    
    def render_session_info(self):
        """Render session information and statistics"""
        
        with st.sidebar:
            st.markdown("### 📊 **Session Info**")
            
            # Session duration
            session_start = st.session_state.get('session_start_time', datetime.now())
            duration = datetime.now() - session_start
            
            st.write(f"**Duration:** {duration.seconds // 60}m {duration.seconds % 60}s")
            
            # Processing statistics
            total_files = st.session_state.get('total_files_processed', 0)
            total_items = st.session_state.get('total_items_processed', 0)
            total_cost = st.session_state.get('total_session_cost', 0.0)
            
            st.write(f"**Files Processed:** {total_files}")
            st.write(f"**Items Enhanced:** {total_items}")
            st.write(f"**Session Cost:** ${total_cost:.4f}")
            
            # Memory usage (if available)
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                st.write(f"**Memory Usage:** {memory_mb:.1f} MB")
            except:
                pass
    
    def render_keyboard_shortcuts(self):
        """Render keyboard shortcuts help"""
        
        with st.expander("⌨️ Keyboard Shortcuts", expanded=False):
            st.markdown("""
            **Navigation:**
            - `Ctrl + R` - Refresh page
            - `Ctrl + Shift + R` - Hard refresh (clear cache)
            
            **Editing:**
            - `Ctrl + A` - Select all text
            - `Ctrl + Z` - Undo
            - `Ctrl + Y` - Redo
            
            **Application:**
            - `F11` - Toggle fullscreen
            - `Ctrl + +/-` - Zoom in/out
            """)


# Global UI polish instance
ui_polish = EnhancedUIPolish()

