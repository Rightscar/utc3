"""
Enhanced Theming Module
=======================

Implements optional theming functionality for the Universal AI Training Data Creator.
Provides multiple color schemes and UI customization options.

Features:
- Multiple pre-built themes
- Dark/Light mode toggle
- Custom color schemes
- Font size adjustments
- Layout preferences
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTheming:
    """Enhanced Theming System - Optional Add-on"""
    
    def __init__(self):
        self.themes = {
            'modern_gradient': {
                'name': '🌈 Modern Gradient',
                'primary_color': '#667eea',
                'background_color': '#FFFFFF',
                'secondary_background_color': '#f7fafc',
                'text_color': '#2d3748',
                'font': 'Inter, system-ui, sans-serif',
                'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
            },
            'neon_cyber': {
                'name': '⚡ Neon Cyber',
                'primary_color': '#00f5ff',
                'background_color': '#0a0e27',
                'secondary_background_color': '#1a1f3a',
                'text_color': '#ffffff',
                'font': 'JetBrains Mono, monospace',
                'gradient': 'linear-gradient(135deg, #00f5ff 0%, #ff00ff 100%)'
            },
            'glass_morphism': {
                'name': '🔮 Glass Morphism',
                'primary_color': '#8b5cf6',
                'background_color': '#f8fafc',
                'secondary_background_color': 'rgba(255, 255, 255, 0.25)',
                'text_color': '#1e293b',
                'font': 'SF Pro Display, system-ui, sans-serif',
                'gradient': 'linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%)'
            },
            'aurora': {
                'name': '🌌 Aurora',
                'primary_color': '#a855f7',
                'background_color': '#0f0f23',
                'secondary_background_color': '#1a1a2e',
                'text_color': '#e2e8f0',
                'font': 'Poppins, sans-serif',
                'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)'
            },
            'minimal_zen': {
                'name': '🎋 Minimal Zen',
                'primary_color': '#059669',
                'background_color': '#fefefe',
                'secondary_background_color': '#f0fdf4',
                'text_color': '#064e3b',
                'font': 'Inter, system-ui, sans-serif',
                'gradient': 'linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)'
            },
            'sunset_vibes': {
                'name': '🌅 Sunset Vibes',
                'primary_color': '#f59e0b',
                'background_color': '#fffbeb',
                'secondary_background_color': '#fef3c7',
                'text_color': '#92400e',
                'font': 'Nunito, sans-serif',
                'gradient': 'linear-gradient(135deg, #fbbf24 0%, #f59e0b 50%, #dc2626 100%)'
            },
            'ocean_depth': {
                'name': '🌊 Ocean Depth',
                'primary_color': '#0ea5e9',
                'background_color': '#f0f9ff',
                'secondary_background_color': '#e0f2fe',
                'text_color': '#0c4a6e',
                'font': 'Roboto, sans-serif',
                'gradient': 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 50%, #0369a1 100%)'
            },
            'retro_wave': {
                'name': '🕹️ Retro Wave',
                'primary_color': '#ff0080',
                'background_color': '#1a0033',
                'secondary_background_color': '#2d1b69',
                'text_color': '#ffffff',
                'font': 'Orbitron, monospace',
                'gradient': 'linear-gradient(135deg, #ff0080 0%, #7928ca 50%, #0070f3 100%)'
            }
        }
        
        # Initialize session state
        if 'selected_theme' not in st.session_state:
            st.session_state.selected_theme = 'modern_gradient'
        
        if 'custom_font_size' not in st.session_state:
            st.session_state.custom_font_size = 16
        
        if 'sidebar_width' not in st.session_state:
            st.session_state.sidebar_width = 'normal'
    
    def render_theme_selector(self) -> None:
        """Render theme selection UI in sidebar"""
        
        with st.sidebar:
            st.markdown("### 🎨 **Theme Settings**")
            
            # Theme selection
            theme_options = [f"{theme['name']}" for theme in self.themes.values()]
            theme_keys = list(self.themes.keys())
            
            current_index = theme_keys.index(st.session_state.selected_theme) if st.session_state.selected_theme in theme_keys else 0
            
            selected_theme_name = st.selectbox(
                "Choose Theme",
                options=theme_options,
                index=current_index,
                key="theme_selector"
            )
            
            # Update selected theme
            selected_key = theme_keys[theme_options.index(selected_theme_name)]
            if selected_key != st.session_state.selected_theme:
                st.session_state.selected_theme = selected_key
                st.rerun()
            
            # Font size adjustment
            st.session_state.custom_font_size = st.slider(
                "Font Size",
                min_value=12,
                max_value=24,
                value=st.session_state.custom_font_size,
                step=1,
                help="Adjust the base font size for better readability"
            )
            
            # Layout preferences
            st.session_state.sidebar_width = st.selectbox(
                "Sidebar Width",
                options=['narrow', 'normal', 'wide'],
                index=['narrow', 'normal', 'wide'].index(st.session_state.sidebar_width),
                help="Adjust sidebar width for your preference"
            )
            
            # Theme preview
            self._render_theme_preview()
            
            st.markdown("---")
    
    def _render_theme_preview(self) -> None:
        """Render a small preview of the selected theme"""
        
        current_theme = self.themes[st.session_state.selected_theme]
        
        st.markdown("#### 👀 **Preview**")
        
        # Create preview using HTML/CSS
        preview_html = f"""
        <div style="
            background-color: {current_theme['background_color']};
            color: {current_theme['text_color']};
            padding: 12px;
            border-radius: 8px;
            border: 2px solid {current_theme['primary_color']};
            font-family: {current_theme['font']};
            font-size: {st.session_state.custom_font_size}px;
            margin: 8px 0;
        ">
            <div style="color: {current_theme['primary_color']}; font-weight: bold; margin-bottom: 4px;">
                Sample Header
            </div>
            <div style="
                background-color: {current_theme['secondary_background_color']};
                padding: 8px;
                border-radius: 4px;
                margin: 4px 0;
            ">
                Sample content with background
            </div>
            <div style="font-size: {st.session_state.custom_font_size - 2}px; opacity: 0.8;">
                Secondary text example
            </div>
        </div>
        """
        
        st.markdown(preview_html, unsafe_allow_html=True)
    
    def apply_theme(self) -> None:
        """Apply the selected theme to the Streamlit app"""
        
        current_theme = self.themes[st.session_state.selected_theme]
        
        # Generate CSS for the theme
        theme_css = self._generate_theme_css(current_theme)
        
        # Apply the CSS
        st.markdown(theme_css, unsafe_allow_html=True)
    
    def _generate_theme_css(self, theme: Dict[str, str]) -> str:
        """Generate CSS for the given theme"""
        
        # Sidebar width mapping
        sidebar_widths = {
            'narrow': '250px',
            'normal': '300px',
            'wide': '350px'
        }
        
        sidebar_width = sidebar_widths.get(st.session_state.sidebar_width, '300px')
        
        css = f"""
        <style>
        /* Main theme colors */
        .stApp {{
            background-color: {theme['background_color']};
            color: {theme['text_color']};
            font-family: {theme['font']};
            font-size: {st.session_state.custom_font_size}px;
        }}
        
        /* Sidebar styling */
        .css-1d391kg {{
            background-color: {theme['secondary_background_color']};
            width: {sidebar_width} !important;
        }}
        
        /* Primary buttons and elements */
        .stButton > button {{
            background-color: {theme['primary_color']};
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            background-color: {theme['primary_color']}dd;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Selectbox and input styling */
        .stSelectbox > div > div {{
            background-color: {theme['secondary_background_color']};
            border: 1px solid {theme['primary_color']}40;
        }}
        
        .stTextInput > div > div > input {{
            background-color: {theme['secondary_background_color']};
            border: 1px solid {theme['primary_color']}40;
            color: {theme['text_color']};
        }}
        
        /* File uploader */
        .stFileUploader > div {{
            background-color: {theme['secondary_background_color']};
            border: 2px dashed {theme['primary_color']}60;
            border-radius: 8px;
        }}
        
        /* Metrics styling */
        .metric-container {{
            background-color: {theme['secondary_background_color']};
            padding: 12px;
            border-radius: 8px;
            border-left: 4px solid {theme['primary_color']};
        }}
        
        /* Success/Info/Warning/Error messages */
        .stSuccess {{
            background-color: #10B98140;
            border: 1px solid #10B981;
            color: #064E3B;
        }}
        
        .stInfo {{
            background-color: {theme['primary_color']}20;
            border: 1px solid {theme['primary_color']};
            color: {theme['text_color']};
        }}
        
        .stWarning {{
            background-color: #F59E0B40;
            border: 1px solid #F59E0B;
            color: #92400E;
        }}
        
        .stError {{
            background-color: #EF444440;
            border: 1px solid #EF4444;
            color: #991B1B;
        }}
        
        /* Expander styling */
        .streamlit-expanderHeader {{
            background-color: {theme['secondary_background_color']};
            border: 1px solid {theme['primary_color']}40;
            border-radius: 8px;
        }}
        
        /* Progress bar */
        .stProgress > div > div > div {{
            background-color: {theme['primary_color']};
        }}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: {theme['secondary_background_color']};
            border-radius: 8px;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            color: {theme['text_color']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {theme['primary_color']};
            color: white;
        }}
        
        /* Custom styling for enhanced components */
        .enhanced-card {{
            background-color: {theme['secondary_background_color']};
            padding: 16px;
            border-radius: 12px;
            border: 1px solid {theme['primary_color']}30;
            margin: 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        
        .enhanced-header {{
            color: {theme['primary_color']};
            font-weight: 600;
            font-size: {st.session_state.custom_font_size + 4}px;
            margin-bottom: 8px;
        }}
        
        .enhanced-subheader {{
            color: {theme['text_color']};
            font-weight: 500;
            font-size: {st.session_state.custom_font_size + 2}px;
            margin-bottom: 6px;
        }}
        
        /* Comparison viewer styling */
        .comparison-raw {{
            background-color: #F8F9FA;
            border-left: 4px solid #6C757D;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
        }}
        
        .comparison-enhanced {{
            background-color: #E8F5E8;
            border-left: 4px solid #28A745;
            padding: 12px;
            border-radius: 8px;
            margin: 8px 0;
            box-shadow: 0 2px 4px rgba(40, 167, 69, 0.1);
        }}
        
        /* Manual review styling */
        .review-item {{
            background-color: {theme['secondary_background_color']};
            border: 1px solid {theme['primary_color']}30;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
        }}
        
        .review-approved {{
            border-left: 4px solid #10B981;
        }}
        
        .review-rejected {{
            border-left: 4px solid #EF4444;
        }}
        
        /* Responsive design */
        @media (max-width: 768px) {{
            .css-1d391kg {{
                width: 250px !important;
            }}
            
            .stApp {{
                font-size: {max(st.session_state.custom_font_size - 2, 12)}px;
            }}
        }}
        </style>
        """
        
        return css
    
    def get_current_theme(self) -> Dict[str, str]:
        """Get the currently selected theme configuration"""
        return self.themes[st.session_state.selected_theme]
    
    def export_theme_config(self) -> Dict[str, Any]:
        """Export current theme configuration for saving/sharing"""
        return {
            'theme': st.session_state.selected_theme,
            'font_size': st.session_state.custom_font_size,
            'sidebar_width': st.session_state.sidebar_width,
            'timestamp': datetime.now().isoformat()
        }
    
    def import_theme_config(self, config: Dict[str, Any]) -> bool:
        """Import theme configuration from saved settings"""
        try:
            if config.get('theme') in self.themes:
                st.session_state.selected_theme = config['theme']
            
            if 'font_size' in config:
                st.session_state.custom_font_size = max(12, min(24, config['font_size']))
            
            if 'sidebar_width' in config and config['sidebar_width'] in ['narrow', 'normal', 'wide']:
                st.session_state.sidebar_width = config['sidebar_width']
            
            return True
        except Exception as e:
            logger.error(f"Error importing theme config: {e}")
            return False
    
    def render_theme_export_import(self) -> None:
        """Render theme export/import functionality"""
        
        with st.sidebar:
            st.markdown("#### 💾 **Theme Settings**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📤 Export", help="Export current theme settings"):
                    config = self.export_theme_config()
                    st.download_button(
                        "💾 Download",
                        data=str(config),
                        file_name=f"theme_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                uploaded_config = st.file_uploader(
                    "📥 Import",
                    type=['json'],
                    help="Import theme settings from file",
                    key="theme_import"
                )
                
                if uploaded_config:
                    try:
                        config = eval(uploaded_config.read().decode())
                        if self.import_theme_config(config):
                            st.success("✅ Theme imported!")
                            st.rerun()
                        else:
                            st.error("❌ Invalid theme file")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    def render_accessibility_options(self) -> None:
        """Render accessibility options"""
        
        with st.sidebar:
            st.markdown("#### ♿ **Accessibility**")
            
            # High contrast mode
            high_contrast = st.checkbox(
                "High Contrast Mode",
                help="Increase contrast for better visibility"
            )
            
            if high_contrast:
                # Apply high contrast CSS
                high_contrast_css = """
                <style>
                .stApp {
                    filter: contrast(1.2) brightness(1.1);
                }
                </style>
                """
                st.markdown(high_contrast_css, unsafe_allow_html=True)
            
            # Large text mode
            large_text = st.checkbox(
                "Large Text Mode",
                help="Increase text size for better readability"
            )
            
            if large_text:
                st.session_state.custom_font_size = max(st.session_state.custom_font_size, 18)
            
            # Reduced motion
            reduced_motion = st.checkbox(
                "Reduced Motion",
                help="Minimize animations and transitions"
            )
            
            if reduced_motion:
                reduced_motion_css = """
                <style>
                * {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                }
                </style>
                """
                st.markdown(reduced_motion_css, unsafe_allow_html=True)

