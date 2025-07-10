"""
Enhanced Tone Manager Module
===========================

Comprehensive tone management system supporting both spiritual and general-purpose
content enhancement tones. Provides organized categorization and easy selection.

Features:
- Spiritual tone categories for consciousness/wisdom content
- General-purpose tones for business, academic, and everyday content
- Organized tone selection with descriptions
- Dynamic prompt template loading
- Tone recommendation system
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import streamlit as st


@dataclass
class ToneDefinition:
    """Definition of a content enhancement tone"""
    id: str
    name: str
    description: str
    category: str
    icon: str
    use_cases: List[str]
    prompt_file: str
    example_output: str = ""


class EnhancedToneManager:
    """Comprehensive tone management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.prompts_dir = "prompts"
        
        # Initialize tone definitions
        self.tone_definitions = self._initialize_tone_definitions()
        
        # Organize by categories
        self.categories = self._organize_by_categories()
    
    def _initialize_tone_definitions(self) -> Dict[str, ToneDefinition]:
        """Initialize all available tone definitions"""
        
        tones = {
            # Spiritual & Wisdom Tones
            "advaita_vedanta": ToneDefinition(
                id="advaita_vedanta",
                name="Advaita Vedanta",
                description="Non-dual wisdom tradition emphasizing Self-inquiry and direct recognition",
                category="Spiritual & Wisdom",
                icon="🕉️",
                use_cases=[
                    "Non-dual spiritual teachings",
                    "Self-inquiry and consciousness exploration",
                    "Vedantic philosophy and wisdom",
                    "Meditation and awareness practices"
                ],
                prompt_file="advaita_vedanta.txt",
                example_output="The Self is beyond all attributes and descriptions. Through inquiry into 'Who am I?', one discovers this unchanging reality..."
            ),
            
            "zen_buddhism": ToneDefinition(
                id="zen_buddhism",
                name="Zen Buddhism",
                description="Direct, simple wisdom emphasizing present moment awareness and ordinary mind",
                category="Spiritual & Wisdom",
                icon="☯️",
                use_cases=[
                    "Zen teachings and koans",
                    "Mindfulness and meditation",
                    "Buddhist philosophy",
                    "Present moment awareness"
                ],
                prompt_file="zen_buddhism.txt",
                example_output="Before enlightenment, chop wood, carry water. After enlightenment, chop wood, carry water..."
            ),
            
            "christian_mysticism": ToneDefinition(
                id="christian_mysticism",
                name="Christian Mysticism",
                description="Devotional and contemplative Christian spiritual tradition",
                category="Spiritual & Wisdom",
                icon="✝️",
                use_cases=[
                    "Christian contemplative practices",
                    "Mystical theology and spirituality",
                    "Prayer and devotional content",
                    "Sacred heart traditions"
                ],
                prompt_file="christian_mysticism.txt",
                example_output="In the depths of silence, the soul meets its Beloved. Here, divine love transforms all seeking..."
            ),
            
            "sufi_mysticism": ToneDefinition(
                id="sufi_mysticism",
                name="Sufi Mysticism",
                description="Poetic, heart-centered Islamic mystical tradition emphasizing divine love",
                category="Spiritual & Wisdom",
                icon="☪️",
                use_cases=[
                    "Sufi poetry and teachings",
                    "Islamic mysticism",
                    "Divine love and devotion",
                    "Whirling and sacred dance"
                ],
                prompt_file="sufi_mysticism.txt",
                example_output="The Beloved is all in all, the lover merely veils Him. When love has no object, then love is perfect..."
            ),
            
            "mindfulness_meditation": ToneDefinition(
                id="mindfulness_meditation",
                name="Mindfulness Meditation",
                description="Present-moment awareness and mindful observation practices",
                category="Spiritual & Wisdom",
                icon="🧘",
                use_cases=[
                    "Mindfulness training",
                    "Meditation instruction",
                    "Stress reduction practices",
                    "Awareness cultivation"
                ],
                prompt_file="mindfulness_meditation.txt",
                example_output="In this moment, there is only this moment. Breathing in, I know I am breathing in..."
            ),
            
            "universal_wisdom": ToneDefinition(
                id="universal_wisdom",
                name="Universal Wisdom",
                description="Inclusive wisdom drawing from multiple spiritual traditions",
                category="Spiritual & Wisdom",
                icon="🌟",
                use_cases=[
                    "Interfaith spiritual content",
                    "Universal principles",
                    "Perennial philosophy",
                    "Cross-cultural wisdom"
                ],
                prompt_file="universal_wisdom.txt",
                example_output="Truth is one, paths are many. All sincere seekers arrive at the same destination..."
            ),
            
            # General Purpose Tones
            "clarity_focused": ToneDefinition(
                id="clarity_focused",
                name="Clarity Focused",
                description="Crystal clear communication that eliminates confusion and enhances understanding",
                category="General Purpose",
                icon="📘",
                use_cases=[
                    "Educational materials",
                    "Training documentation",
                    "User guides and manuals",
                    "Complex concept explanation"
                ],
                prompt_file="clarity_focused.txt",
                example_output="This concept can be understood in three simple steps: First, identify the core principle..."
            ),
            
            "plain_style": ToneDefinition(
                id="plain_style",
                name="Plain Style",
                description="Straightforward, no-nonsense communication using everyday language",
                category="General Purpose",
                icon="✍️",
                use_cases=[
                    "Public communications",
                    "Government documents",
                    "General audience content",
                    "Accessibility-focused writing"
                ],
                prompt_file="plain_style.txt",
                example_output="Here's what you need to know. This process has three steps. First, you..."
            ),
            
            "analytical": ToneDefinition(
                id="analytical",
                name="Analytical",
                description="Logical, evidence-based analysis with systematic reasoning",
                category="General Purpose",
                icon="🔎",
                use_cases=[
                    "Research reports",
                    "Data analysis",
                    "Critical thinking exercises",
                    "Academic analysis"
                ],
                prompt_file="analytical.txt",
                example_output="Analysis of this data reveals three key patterns. First, the correlation between..."
            ),
            
            "professional_business": ToneDefinition(
                id="professional_business",
                name="Professional/Business",
                description="Polished business communication with strategic focus and executive perspective",
                category="General Purpose",
                icon="💬",
                use_cases=[
                    "Business presentations",
                    "Corporate training",
                    "Executive communications",
                    "Professional development"
                ],
                prompt_file="professional_business.txt",
                example_output="This strategic initiative delivers measurable ROI through three key value drivers..."
            ),
            
            # Additional General Tones
            "conversational_friendly": ToneDefinition(
                id="conversational_friendly",
                name="Conversational & Friendly",
                description="Warm, approachable communication that feels like talking with a helpful friend",
                category="General Purpose",
                icon="😊",
                use_cases=[
                    "Customer support content",
                    "Friendly tutorials",
                    "Community communications",
                    "Approachable explanations"
                ],
                prompt_file="conversational_friendly.txt",
                example_output="Let me walk you through this - it's actually pretty straightforward once you see how it works..."
            ),
            
            "academic_scholarly": ToneDefinition(
                id="academic_scholarly",
                name="Academic & Scholarly",
                description="Rigorous academic tone with scholarly depth and intellectual precision",
                category="Academic & Research",
                icon="🎓",
                use_cases=[
                    "Academic papers",
                    "Research documentation",
                    "Scholarly analysis",
                    "Educational content"
                ],
                prompt_file="academic_scholarly.txt",
                example_output="This phenomenon can be understood through the theoretical framework of..."
            ),
            
            "technical_precise": ToneDefinition(
                id="technical_precise",
                name="Technical & Precise",
                description="Exact technical communication with comprehensive detail and accuracy",
                category="Technical & Professional",
                icon="⚙️",
                use_cases=[
                    "Technical documentation",
                    "API documentation",
                    "Engineering guides",
                    "Specification documents"
                ],
                prompt_file="technical_precise.txt",
                example_output="The implementation requires the following parameters: timeout (integer, 30-300 seconds)..."
            ),
            
            "creative_engaging": ToneDefinition(
                id="creative_engaging",
                name="Creative & Engaging",
                description="Creative, inspiring communication that captivates and motivates",
                category="Creative & Marketing",
                icon="🎨",
                use_cases=[
                    "Marketing content",
                    "Creative writing",
                    "Inspirational materials",
                    "Storytelling content"
                ],
                prompt_file="creative_engaging.txt",
                example_output="Imagine a world where this challenge becomes your greatest opportunity..."
            )
        }
        
        return tones
    
    def _organize_by_categories(self) -> Dict[str, List[str]]:
        """Organize tones by categories"""
        
        categories = {}
        for tone_id, tone_def in self.tone_definitions.items():
            category = tone_def.category
            if category not in categories:
                categories[category] = []
            categories[category].append(tone_id)
        
        return categories
    
    def get_tone_definition(self, tone_id: str) -> Optional[ToneDefinition]:
        """Get tone definition by ID"""
        return self.tone_definitions.get(tone_id)
    
    def get_tones_by_category(self, category: str) -> List[ToneDefinition]:
        """Get all tones in a specific category"""
        tone_ids = self.categories.get(category, [])
        return [self.tone_definitions[tone_id] for tone_id in tone_ids]
    
    def get_all_categories(self) -> List[str]:
        """Get all available categories"""
        return list(self.categories.keys())
    
    def get_available_tones(self) -> List[str]:
        """Get all available tone IDs"""
        return list(self.tone_definitions.keys())
    
    def load_prompt_template(self, tone_id: str) -> Optional[str]:
        """Load prompt template for a specific tone"""
        
        tone_def = self.get_tone_definition(tone_id)
        if not tone_def:
            self.logger.error(f"Tone definition not found: {tone_id}")
            return None
        
        prompt_path = os.path.join(self.prompts_dir, tone_def.prompt_file)
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            self.logger.error(f"Prompt file not found: {prompt_path}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading prompt template: {e}")
            return None
    
    def recommend_tone(self, content_type: str, audience: str, purpose: str) -> List[str]:
        """Recommend appropriate tones based on content characteristics"""
        
        recommendations = []
        
        # Content type recommendations
        if "spiritual" in content_type.lower() or "wisdom" in content_type.lower():
            recommendations.extend(["universal_wisdom", "mindfulness_meditation"])
        elif "business" in content_type.lower() or "corporate" in content_type.lower():
            recommendations.extend(["professional_business", "analytical"])
        elif "technical" in content_type.lower() or "documentation" in content_type.lower():
            recommendations.extend(["technical_precise", "clarity_focused"])
        elif "academic" in content_type.lower() or "research" in content_type.lower():
            recommendations.extend(["academic_scholarly", "analytical"])
        elif "creative" in content_type.lower() or "marketing" in content_type.lower():
            recommendations.extend(["creative_engaging", "conversational_friendly"])
        
        # Audience recommendations
        if "general" in audience.lower() or "public" in audience.lower():
            recommendations.extend(["plain_style", "clarity_focused"])
        elif "professional" in audience.lower() or "executive" in audience.lower():
            recommendations.extend(["professional_business", "analytical"])
        elif "academic" in audience.lower() or "scholar" in audience.lower():
            recommendations.extend(["academic_scholarly", "technical_precise"])
        elif "student" in audience.lower() or "learner" in audience.lower():
            recommendations.extend(["clarity_focused", "conversational_friendly"])
        
        # Purpose recommendations
        if "training" in purpose.lower() or "education" in purpose.lower():
            recommendations.extend(["clarity_focused", "conversational_friendly"])
        elif "analysis" in purpose.lower() or "research" in purpose.lower():
            recommendations.extend(["analytical", "academic_scholarly"])
        elif "inspiration" in purpose.lower() or "motivation" in purpose.lower():
            recommendations.extend(["creative_engaging", "universal_wisdom"])
        elif "instruction" in purpose.lower() or "guide" in purpose.lower():
            recommendations.extend(["plain_style", "technical_precise"])
        
        # Remove duplicates and return top recommendations
        unique_recommendations = list(dict.fromkeys(recommendations))
        return unique_recommendations[:3]  # Return top 3 recommendations
    
    def render_tone_selector(self, default_tone: str = "clarity_focused") -> str:
        """Render tone selection interface in Streamlit"""
        
        st.subheader("🎭 Select Enhancement Tone")
        
        # Category tabs
        categories = self.get_all_categories()
        category_tabs = st.tabs(categories)
        
        selected_tone = default_tone
        
        for i, category in enumerate(categories):
            with category_tabs[i]:
                tones = self.get_tones_by_category(category)
                
                for tone in tones:
                    col1, col2 = st.columns([1, 4])
                    
                    with col1:
                        if st.button(f"{tone.icon} Select", key=f"select_{tone.id}"):
                            selected_tone = tone.id
                            st.session_state['selected_tone'] = selected_tone
                    
                    with col2:
                        st.write(f"**{tone.name}**")
                        st.write(tone.description)
                        
                        with st.expander("Details & Use Cases"):
                            st.write("**Best for:**")
                            for use_case in tone.use_cases:
                                st.write(f"• {use_case}")
                            
                            if tone.example_output:
                                st.write("**Example output style:**")
                                st.write(f"*{tone.example_output}*")
        
        # Show current selection
        current_tone = st.session_state.get('selected_tone', selected_tone)
        current_def = self.get_tone_definition(current_tone)
        
        if current_def:
            st.success(f"**Selected:** {current_def.icon} {current_def.name}")
            st.info(current_def.description)
        
        return current_tone
    
    def render_tone_recommendation_widget(self):
        """Render tone recommendation widget"""
        
        with st.expander("🎯 Get Tone Recommendations", expanded=False):
            st.write("Answer a few questions to get personalized tone recommendations:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                content_type = st.selectbox(
                    "Content Type",
                    ["General", "Business", "Technical", "Academic", "Creative", "Spiritual", "Educational"]
                )
            
            with col2:
                audience = st.selectbox(
                    "Target Audience",
                    ["General Public", "Professionals", "Students", "Executives", "Academics", "Technical Users"]
                )
            
            with col3:
                purpose = st.selectbox(
                    "Primary Purpose",
                    ["Education", "Training", "Analysis", "Inspiration", "Documentation", "Communication"]
                )
            
            if st.button("Get Recommendations"):
                recommendations = self.recommend_tone(content_type, audience, purpose)
                
                if recommendations:
                    st.write("**Recommended tones for your content:**")
                    for tone_id in recommendations:
                        tone_def = self.get_tone_definition(tone_id)
                        if tone_def:
                            st.write(f"• {tone_def.icon} **{tone_def.name}**: {tone_def.description}")
                else:
                    st.write("Consider starting with **Clarity Focused** for general content enhancement.")
    
    def get_tone_statistics(self) -> Dict[str, Any]:
        """Get statistics about available tones"""
        
        total_tones = len(self.tone_definitions)
        categories_count = len(self.categories)
        
        category_distribution = {}
        for category, tone_ids in self.categories.items():
            category_distribution[category] = len(tone_ids)
        
        return {
            "total_tones": total_tones,
            "categories_count": categories_count,
            "category_distribution": category_distribution,
            "spiritual_tones": len(self.get_tones_by_category("Spiritual & Wisdom")),
            "general_tones": total_tones - len(self.get_tones_by_category("Spiritual & Wisdom"))
        }


# Global enhanced tone manager instance
enhanced_tone_manager = EnhancedToneManager()


def get_enhanced_tone_manager() -> EnhancedToneManager:
    """Get the global enhanced tone manager instance"""
    return enhanced_tone_manager

