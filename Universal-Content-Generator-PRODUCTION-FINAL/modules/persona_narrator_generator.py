"""
Custom Persona Narrator Generator - Core Use Case 4
==================================================

Transforms any content as if narrated by famous personalities, historical figures,
or custom personas. Creates immersive storytelling experiences.

Features:
- Famous personalities (Einstein, Ramana Maharshi, Elon Musk, etc.)
- Historical figures (Shakespeare, Gandhi, Tesla, etc.)
- Custom persona creation
- Voice consistency and authenticity
- Narrative style adaptation
"""

import streamlit as st
import logging
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PersonaNarrationResult:
    """Represents a persona narration result"""
    id: str
    original_chunk_id: str
    persona_name: str
    narrated_content: str
    persona_traits: List[str]
    authenticity_score: float = 0.8
    narrative_style: str = "storytelling"
    word_count: int = 0

class PersonaNarratorGenerator:
    """Generate content narrated by specific personas"""
    
    def __init__(self):
        self.persona_library = self._load_persona_library()
        self.narrative_styles = self._load_narrative_styles()
        
    def _load_persona_library(self) -> Dict[str, Dict[str, Any]]:
        """Load library of famous personas and their characteristics"""
        return {
            # Spiritual/Philosophical Figures
            "ramana_maharshi": {
                "name": "Ramana Maharshi",
                "category": "Spiritual Teacher",
                "description": "Enlightened sage known for self-inquiry and direct pointing to truth",
                "key_traits": [
                    "Speaks with gentle authority",
                    "Uses simple, direct language",
                    "Often asks 'Who am I?' type questions",
                    "Points to the Self/Awareness",
                    "Compassionate and patient tone"
                ],
                "speaking_style": "gentle, direct, questioning",
                "famous_phrases": ["Who am I?", "Be as you are", "The Self is ever-present"],
                "prompt_template": """
Narrate this content as Ramana Maharshi would tell it. Use his gentle, direct style of pointing to truth through self-inquiry.

KEY CHARACTERISTICS:
- Gentle but authoritative voice
- Simple, clear language
- Often uses questions to guide understanding
- Points to the ever-present Self/Awareness
- Compassionate and patient
- Uses analogies from nature and daily life

CONTENT TO NARRATE:
{content}

Narrate this as Ramana would, maintaining his authentic voice and wisdom approach.
"""
            },
            
            "einstein": {
                "name": "Albert Einstein",
                "category": "Scientist/Philosopher",
                "description": "Brilliant physicist known for relativity and deep philosophical insights",
                "key_traits": [
                    "Curious and wonder-filled",
                    "Uses thought experiments",
                    "Connects science with philosophy",
                    "Humble about the mysteries of existence",
                    "Playful yet profound"
                ],
                "speaking_style": "curious, thoughtful, wonder-filled",
                "famous_phrases": ["Imagination is more important than knowledge", "God does not play dice", "The most beautiful thing we can experience is the mysterious"],
                "prompt_template": """
Narrate this content as Albert Einstein would tell it. Use his curious, wonder-filled approach to understanding reality.

KEY CHARACTERISTICS:
- Deep curiosity and sense of wonder
- Uses thought experiments and analogies
- Connects scientific thinking with philosophical insights
- Humble about the mysteries of existence
- Playful yet profound approach
- Often references the beauty and mystery of the universe

CONTENT TO NARRATE:
{content}

Narrate this as Einstein would, with his characteristic curiosity and philosophical depth.
"""
            },
            
            "elon_musk": {
                "name": "Elon Musk",
                "category": "Entrepreneur/Innovator",
                "description": "Visionary entrepreneur focused on advancing human civilization",
                "key_traits": [
                    "Future-focused and optimistic",
                    "Systems thinking approach",
                    "Direct and sometimes blunt",
                    "Focuses on solving big problems",
                    "Uses first principles reasoning"
                ],
                "speaking_style": "direct, future-focused, systematic",
                "famous_phrases": ["First principles thinking", "Making life multiplanetary", "Accelerating sustainable transport"],
                "prompt_template": """
Narrate this content as Elon Musk would tell it. Use his direct, future-focused approach and systems thinking.

KEY CHARACTERISTICS:
- Future-focused and optimistic about human potential
- Uses first principles reasoning
- Direct and sometimes blunt communication
- Focuses on solving civilization-scale problems
- Systems thinking approach
- Often connects ideas to advancing humanity

CONTENT TO NARRATE:
{content}

Narrate this as Elon would, with his characteristic directness and future-oriented perspective.
"""
            },
            
            "shakespeare": {
                "name": "William Shakespeare",
                "category": "Playwright/Poet",
                "description": "Master of language and human nature",
                "key_traits": [
                    "Rich, poetic language",
                    "Deep insights into human nature",
                    "Uses metaphors and imagery",
                    "Dramatic and theatrical",
                    "Timeless wisdom"
                ],
                "speaking_style": "poetic, dramatic, metaphorical",
                "famous_phrases": ["To be or not to be", "All the world's a stage", "What's in a name?"],
                "prompt_template": """
Narrate this content as William Shakespeare would tell it. Use his rich, poetic language and deep insights into human nature.

KEY CHARACTERISTICS:
- Rich, poetic language with metaphors and imagery
- Deep understanding of human nature and emotions
- Dramatic and theatrical presentation
- Uses analogies from nature, theater, and life
- Timeless wisdom expressed beautifully
- Occasionally uses Elizabethan expressions

CONTENT TO NARRATE:
{content}

Narrate this as Shakespeare would, with his characteristic poetry and insight into the human condition.
"""
            },
            
            "gandhi": {
                "name": "Mahatma Gandhi",
                "category": "Spiritual/Political Leader",
                "description": "Advocate of non-violence and truth",
                "key_traits": [
                    "Gentle but firm conviction",
                    "Emphasizes truth and non-violence",
                    "Simple, humble language",
                    "Connects personal and social transformation",
                    "Practical wisdom"
                ],
                "speaking_style": "gentle, firm, truthful",
                "famous_phrases": ["Be the change you wish to see", "Truth is God", "Non-violence is the greatest force"],
                "prompt_template": """
Narrate this content as Mahatma Gandhi would tell it. Use his gentle but firm approach to truth and transformation.

KEY CHARACTERISTICS:
- Gentle but unwavering conviction
- Emphasizes truth (satya) and non-violence (ahimsa)
- Simple, humble language accessible to all
- Connects personal transformation with social change
- Practical wisdom for daily life
- Compassionate understanding of human struggles

CONTENT TO NARRATE:
{content}

Narrate this as Gandhi would, with his characteristic gentleness and commitment to truth.
"""
            },
            
            "steve_jobs": {
                "name": "Steve Jobs",
                "category": "Innovator/Designer",
                "description": "Perfectionist focused on beautiful, simple design",
                "key_traits": [
                    "Obsessed with simplicity and elegance",
                    "Perfectionist attention to detail",
                    "Passionate and intense",
                    "Focuses on user experience",
                    "Thinks different"
                ],
                "speaking_style": "passionate, precise, elegant",
                "famous_phrases": ["Think different", "Simplicity is the ultimate sophistication", "Stay hungry, stay foolish"],
                "prompt_template": """
Narrate this content as Steve Jobs would tell it. Use his passionate focus on simplicity, elegance, and thinking differently.

KEY CHARACTERISTICS:
- Obsessed with simplicity and elegant design
- Perfectionist attention to every detail
- Passionate and intense about quality
- Focuses on the user experience and human needs
- Challenges conventional thinking
- Believes in the power of beautiful, simple solutions

CONTENT TO NARRATE:
{content}

Narrate this as Steve Jobs would, with his characteristic passion for simplicity and excellence.
"""
            },
            
            "rumi": {
                "name": "Rumi",
                "category": "Mystic Poet",
                "description": "Sufi mystic known for ecstatic poetry about divine love",
                "key_traits": [
                    "Ecstatic and passionate",
                    "Uses metaphors of love and longing",
                    "Mystical and transcendent",
                    "Poetic and flowing language",
                    "Speaks of unity and divine love"
                ],
                "speaking_style": "ecstatic, poetic, mystical",
                "famous_phrases": ["Let yourself be silently drawn", "The wound is the place where the Light enters", "Love is the bridge"],
                "prompt_template": """
Narrate this content as Rumi would tell it. Use his ecstatic, poetic style filled with mystical love and longing.

KEY CHARACTERISTICS:
- Ecstatic and passionate about divine love
- Uses rich metaphors of love, longing, and union
- Mystical and transcendent perspective
- Flowing, poetic language
- Speaks of the beloved, unity, and spiritual intoxication
- Often uses imagery of wine, dancing, and fire

CONTENT TO NARRATE:
{content}

Narrate this as Rumi would, with his characteristic ecstasy and mystical poetry.
"""
            },
            
            "carl_sagan": {
                "name": "Carl Sagan",
                "category": "Scientist/Educator",
                "description": "Astronomer who made science accessible and inspiring",
                "key_traits": [
                    "Sense of wonder about the cosmos",
                    "Makes complex ideas accessible",
                    "Poetic about scientific discoveries",
                    "Humble before the vastness of space",
                    "Optimistic about human potential"
                ],
                "speaking_style": "wonder-filled, accessible, poetic",
                "famous_phrases": ["Cosmos", "Billions and billions", "We are made of star stuff"],
                "prompt_template": """
Narrate this content as Carl Sagan would tell it. Use his sense of cosmic wonder and gift for making complex ideas accessible.

KEY CHARACTERISTICS:
- Deep sense of wonder about the cosmos and our place in it
- Makes complex scientific ideas accessible to everyone
- Poetic and inspiring about scientific discoveries
- Humble before the vastness and mystery of the universe
- Optimistic about human potential and our cosmic journey
- Often connects human experience to cosmic perspective

CONTENT TO NARRATE:
{content}

Narrate this as Carl Sagan would, with his characteristic wonder and cosmic perspective.
"""
            }
        }
    
    def _load_narrative_styles(self) -> Dict[str, str]:
        """Load different narrative styles"""
        return {
            "storytelling": "Tell it as an engaging story with narrative flow",
            "teaching": "Present it as a wise teacher sharing knowledge",
            "conversation": "Share it as if in intimate conversation",
            "lecture": "Present it as an inspiring lecture or talk",
            "reflection": "Share it as personal reflection and insight",
            "dialogue": "Present it as dialogue with the audience",
            "memoir": "Tell it as if recounting personal experience"
        }
    
    def get_available_personas(self) -> Dict[str, Dict[str, Any]]:
        """Get all available personas"""
        return self.persona_library
    
    def get_narrative_styles(self) -> Dict[str, str]:
        """Get available narrative styles"""
        return self.narrative_styles
    
    def get_personas_by_category(self) -> Dict[str, List[str]]:
        """Get personas organized by category"""
        categories = {}
        for persona_id, persona_info in self.persona_library.items():
            category = persona_info["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(persona_id)
        return categories
    
    def generate_persona_narration(self, chunks: List[Dict[str, Any]],
                                 persona: str,
                                 narrative_style: str = "storytelling",
                                 custom_persona_description: str = None) -> List[PersonaNarrationResult]:
        """
        Generate content narrated by a specific persona
        
        Args:
            chunks: List of content chunks
            persona: Persona identifier or custom persona name
            narrative_style: Style of narration
            custom_persona_description: Description for custom personas
            
        Returns:
            List of PersonaNarrationResult objects
        """
        results = []
        
        for chunk in chunks:
            try:
                narration_result = self._generate_chunk_persona_narration(
                    chunk, persona, narrative_style, custom_persona_description
                )
                results.append(narration_result)
                
                # Add small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error generating persona narration for chunk {chunk.get('id', 'unknown')}: {e}")
                # Create fallback narration
                fallback_result = self._create_fallback_persona_narration(
                    chunk, persona, narrative_style
                )
                results.append(fallback_result)
        
        return results
    
    def _generate_chunk_persona_narration(self, chunk: Dict[str, Any],
                                        persona: str,
                                        narrative_style: str,
                                        custom_persona_description: str = None) -> PersonaNarrationResult:
        """Generate persona narration for a single chunk"""
        
        # Create prompt based on persona type
        if persona in self.persona_library:
            prompt = self._create_known_persona_prompt(chunk, persona, narrative_style)
            persona_name = self.persona_library[persona]["name"]
            persona_traits = self.persona_library[persona]["key_traits"]
        else:
            prompt = self._create_custom_persona_prompt(
                chunk, persona, narrative_style, custom_persona_description
            )
            persona_name = persona
            persona_traits = ["Custom persona characteristics"]
        
        try:
            # Import OpenAI here to handle missing dependency gracefully
            import openai
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {persona_name}. Narrate content in your authentic voice, maintaining your characteristic style, wisdom, and personality throughout."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            # Extract narrated content
            narrated_content = response.choices[0].message.content.strip()
            
            # Create result object
            result = PersonaNarrationResult(
                id=f"{chunk['id']}_persona_{persona}",
                original_chunk_id=chunk['id'],
                persona_name=persona_name,
                narrated_content=narrated_content,
                persona_traits=persona_traits,
                authenticity_score=0.9,
                narrative_style=narrative_style,
                word_count=len(narrated_content.split())
            )
            
            return result
            
        except ImportError:
            logger.warning("OpenAI not available, using fallback")
            return self._create_fallback_persona_narration(chunk, persona, narrative_style)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._create_fallback_persona_narration(chunk, persona, narrative_style)
    
    def _create_known_persona_prompt(self, chunk: Dict[str, Any],
                                   persona: str,
                                   narrative_style: str) -> str:
        """Create prompt for known persona"""
        
        persona_info = self.persona_library[persona]
        content = chunk['text']
        
        # Use persona's specific prompt template
        base_prompt = persona_info['prompt_template'].format(content=content)
        
        # Add narrative style instruction
        style_instruction = self.narrative_styles[narrative_style]
        
        prompt = f"""
{base_prompt}

NARRATIVE STYLE: {style_instruction}

ADDITIONAL INSTRUCTIONS:
- Maintain {persona_info['name']}'s authentic voice throughout
- Use their characteristic speaking style: {persona_info['speaking_style']}
- Include their typical insights and perspectives
- Make it feel like {persona_info['name']} is personally sharing this with the reader
- Keep the content engaging and true to their personality

NARRATED CONTENT:
"""
        
        return prompt
    
    def _create_custom_persona_prompt(self, chunk: Dict[str, Any],
                                    persona: str,
                                    narrative_style: str,
                                    custom_description: str = None) -> str:
        """Create prompt for custom persona"""
        
        content = chunk['text']
        style_instruction = self.narrative_styles[narrative_style]
        
        if custom_description:
            persona_description = custom_description
        else:
            persona_description = f"A unique persona named {persona} with their own distinctive voice and perspective"
        
        prompt = f"""
Narrate this content as {persona} would tell it.

PERSONA DESCRIPTION:
{persona_description}

CONTENT TO NARRATE:
{content}

NARRATIVE STYLE: {style_instruction}

INSTRUCTIONS:
- Create an authentic voice for {persona} based on the description
- Maintain consistency in their speaking style and perspective
- Make the narration engaging and personal
- Include insights and observations that fit this persona
- Ensure the content flows naturally in their voice

NARRATED CONTENT:
"""
        
        return prompt
    
    def _create_fallback_persona_narration(self, chunk: Dict[str, Any],
                                         persona: str,
                                         narrative_style: str) -> PersonaNarrationResult:
        """Create fallback persona narration when API fails"""
        
        original_text = chunk['text']
        
        # Simple persona transformation
        if persona == "einstein":
            fallback_text = f"As I contemplate this fascinating subject, I am struck by the wonder of it all. {original_text} How marvelous that we can explore such mysteries with our curious minds!"
        elif persona == "ramana_maharshi":
            fallback_text = f"Let us inquire into this together. {original_text} Who is it that seeks to understand? What is the nature of the one who knows?"
        elif persona == "shakespeare":
            fallback_text = f"Hark! What wisdom doth unfold before us. {original_text} Verily, such knowledge doth illuminate the very essence of our being."
        else:
            fallback_text = f"[In the voice of {persona}]: {original_text}"
        
        return PersonaNarrationResult(
            id=f"{chunk['id']}_persona_{persona}_fallback",
            original_chunk_id=chunk['id'],
            persona_name=persona,
            narrated_content=fallback_text,
            persona_traits=["Fallback characteristics"],
            authenticity_score=0.6,
            narrative_style=narrative_style,
            word_count=len(fallback_text.split())
        )
    
    def create_custom_persona(self, name: str, description: str,
                            key_traits: List[str], speaking_style: str) -> Dict[str, Any]:
        """Create a custom persona definition"""
        
        custom_persona = {
            "name": name,
            "category": "Custom",
            "description": description,
            "key_traits": key_traits,
            "speaking_style": speaking_style,
            "famous_phrases": [],
            "prompt_template": f"""
Narrate this content as {name} would tell it.

PERSONA DESCRIPTION: {description}

KEY TRAITS:
{chr(10).join(f"- {trait}" for trait in key_traits)}

SPEAKING STYLE: {speaking_style}

CONTENT TO NARRATE:
{{content}}

Narrate this as {name} would, maintaining their authentic voice and characteristics.
"""
        }
        
        return custom_persona
    
    def preview_persona_voice(self, persona: str, sample_text: str = None) -> str:
        """Generate a preview of how a persona would narrate content"""
        
        if sample_text is None:
            sample_text = "Knowledge is like a garden that grows more beautiful the more we tend to it with curiosity and care."
        
        sample_chunk = {
            'id': 'preview',
            'text': sample_text
        }
        
        result = self._generate_chunk_persona_narration(
            sample_chunk, persona, "storytelling"
        )
        
        return result.narrated_content
    
    def export_persona_narrations(self, narration_results: List[PersonaNarrationResult],
                                 export_format: str = "json") -> str:
        """Export persona narrations in specified format"""
        
        if export_format.lower() == "json":
            export_data = []
            for result in narration_results:
                export_data.append({
                    "id": result.id,
                    "persona": result.persona_name,
                    "narrative_style": result.narrative_style,
                    "content": result.narrated_content,
                    "authenticity_score": result.authenticity_score,
                    "word_count": result.word_count,
                    "persona_traits": result.persona_traits
                })
            return json.dumps(export_data, indent=2)
        
        elif export_format.lower() == "markdown":
            markdown_content = f"# Persona Narrations\n\n"
            
            current_persona = None
            for result in narration_results:
                if result.persona_name != current_persona:
                    current_persona = result.persona_name
                    markdown_content += f"## Narrated by {current_persona}\n\n"
                
                markdown_content += f"### Content Piece\n\n"
                markdown_content += f"{result.narrated_content}\n\n"
                markdown_content += f"**Style:** {result.narrative_style}  \n"
                markdown_content += f"**Authenticity Score:** {result.authenticity_score:.1f}/1.0  \n"
                markdown_content += f"**Word Count:** {result.word_count}\n\n"
                markdown_content += "---\n\n"
            
            return markdown_content
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def get_persona_recommendations(self, content_type: str = "general") -> List[str]:
        """Get persona recommendations based on content type"""
        
        recommendations = {
            "spiritual": ["ramana_maharshi", "rumi", "gandhi"],
            "scientific": ["einstein", "carl_sagan"],
            "business": ["elon_musk", "steve_jobs"],
            "creative": ["shakespeare", "rumi"],
            "philosophical": ["einstein", "ramana_maharshi", "gandhi"],
            "general": ["einstein", "ramana_maharshi", "shakespeare", "elon_musk"]
        }
        
        return recommendations.get(content_type, recommendations["general"])

