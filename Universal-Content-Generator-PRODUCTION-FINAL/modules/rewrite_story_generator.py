"""
Rewrite Story Like Generator - Core Use Case 1
==============================================

Transforms any content into different voices, tones, and perspectives.
Examples: Rewrite as 3-year-old, pirate, Zen monk, dog, etc.

Features:
- Predefined persona templates
- Custom persona creation
- Tone and style adjustments
- Age-appropriate transformations
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
class RewriteResult:
    """Represents a rewritten content result"""
    id: str
    original_chunk_id: str
    persona: str
    rewritten_text: str
    style_settings: Dict[str, Any]
    quality_score: float = 0.8
    word_count: int = 0

class RewriteStoryGenerator:
    """Generate story rewrites in different voices and styles"""
    
    def __init__(self):
        self.persona_templates = self._load_persona_templates()
        self.style_options = self._load_style_options()
        
    def _load_persona_templates(self) -> Dict[str, Dict[str, str]]:
        """Load predefined persona templates"""
        return {
            # Age-based personas
            "3_year_old": {
                "name": "3-Year-Old Child",
                "description": "Simple words, wonder, excitement",
                "prompt_template": "Rewrite this story as if told by a 3-year-old child. Use simple words, show wonder and excitement, ask lots of questions, and make it playful and innocent.",
                "example_traits": ["Simple vocabulary", "Lots of 'wow!' and 'cool!'", "Questions about everything"]
            },
            "5_year_old": {
                "name": "5-Year-Old Child", 
                "description": "Curious, imaginative, learning",
                "prompt_template": "Rewrite this story as if told by a 5-year-old child. Use slightly more complex words but still simple, show curiosity and imagination, and include child-like observations.",
                "example_traits": ["Curious questions", "Imaginative comparisons", "Learning new words"]
            },
            "teenager": {
                "name": "Teenager",
                "description": "Modern slang, dramatic, relatable",
                "prompt_template": "Rewrite this story as if told by a modern teenager. Use current slang, make it dramatic and relatable to teen experiences, and add emotional intensity.",
                "example_traits": ["Modern slang", "Dramatic expressions", "Relatable emotions"]
            },
            
            # Character-based personas
            "pirate": {
                "name": "Pirate Captain",
                "description": "Adventurous, nautical terms, 'arrr!'",
                "prompt_template": "Rewrite this story as if told by a pirate captain. Use nautical terms, say 'arrr!' and 'matey', make it adventurous and bold, and include pirate expressions.",
                "example_traits": ["Nautical vocabulary", "Pirate expressions", "Adventurous tone"]
            },
            "zen_monk": {
                "name": "Zen Monk",
                "description": "Peaceful, wise, mindful",
                "prompt_template": "Rewrite this story as if told by a Zen monk. Use peaceful and wise language, include mindfulness concepts, speak slowly and thoughtfully, and add spiritual insights.",
                "example_traits": ["Peaceful language", "Mindful observations", "Spiritual wisdom"]
            },
            "dog": {
                "name": "Friendly Dog",
                "description": "Excited, loyal, simple thoughts",
                "prompt_template": "Rewrite this story as if told by a friendly dog. Show excitement about simple things, be loyal and loving, use simple thoughts, and include dog-like observations about smells, sounds, and treats.",
                "example_traits": ["Excited about everything", "Mentions smells and sounds", "Simple, happy thoughts"]
            },
            "cat": {
                "name": "Sophisticated Cat",
                "description": "Aloof, elegant, judgmental",
                "prompt_template": "Rewrite this story as if told by a sophisticated cat. Be slightly aloof and elegant, make judgmental observations, show independence, and include cat-like priorities.",
                "example_traits": ["Elegant language", "Judgmental observations", "Independent attitude"]
            },
            
            # Professional personas
            "scientist": {
                "name": "Curious Scientist",
                "description": "Analytical, hypothesis-driven, precise",
                "prompt_template": "Rewrite this story as if told by a curious scientist. Use analytical language, form hypotheses, be precise and methodical, and include scientific observations.",
                "example_traits": ["Analytical thinking", "Hypothesis formation", "Precise language"]
            },
            "poet": {
                "name": "Romantic Poet",
                "description": "Lyrical, metaphorical, emotional",
                "prompt_template": "Rewrite this story as if told by a romantic poet. Use lyrical and metaphorical language, be emotional and expressive, and include poetic imagery.",
                "example_traits": ["Lyrical language", "Rich metaphors", "Emotional expression"]
            },
            "comedian": {
                "name": "Stand-up Comedian",
                "description": "Funny, observational, timing",
                "prompt_template": "Rewrite this story as if told by a stand-up comedian. Make it funny with observational humor, use good timing, and include comedic insights about everyday situations.",
                "example_traits": ["Observational humor", "Good timing", "Funny insights"]
            },
            
            # Historical/Famous personas
            "shakespeare": {
                "name": "William Shakespeare",
                "description": "Elizabethan English, dramatic, poetic",
                "prompt_template": "Rewrite this story as if told by William Shakespeare. Use Elizabethan English style, be dramatic and poetic, include 'thee' and 'thou', and make it theatrical.",
                "example_traits": ["Elizabethan language", "Dramatic flair", "Poetic expressions"]
            },
            "einstein": {
                "name": "Albert Einstein",
                "description": "Thoughtful, curious, scientific wonder",
                "prompt_template": "Rewrite this story as if told by Albert Einstein. Be thoughtful and curious, show scientific wonder, use thought experiments, and include insights about the nature of reality.",
                "example_traits": ["Scientific curiosity", "Thought experiments", "Wonder about reality"]
            },
            "yoda": {
                "name": "Master Yoda",
                "description": "Wise, backwards speech, Force wisdom",
                "prompt_template": "Rewrite this story as if told by Master Yoda. Use backwards sentence structure sometimes, be wise and mystical, include Force wisdom, and speak in Yoda's unique style.",
                "example_traits": ["Backwards speech patterns", "Mystical wisdom", "Force references"]
            }
        }
    
    def _load_style_options(self) -> Dict[str, List[str]]:
        """Load style configuration options"""
        return {
            "tone": [
                "Playful", "Serious", "Humorous", "Dramatic", 
                "Peaceful", "Excited", "Mysterious", "Friendly"
            ],
            "complexity": [
                "Very Simple", "Simple", "Moderate", "Complex", "Very Complex"
            ],
            "length": [
                "Much Shorter", "Shorter", "Same Length", "Longer", "Much Longer"
            ],
            "focus": [
                "Keep Original Meaning", "Add Personality", "Emphasize Emotion", 
                "Add Humor", "Add Wisdom", "Make Educational"
            ]
        }
    
    def get_available_personas(self) -> Dict[str, Dict[str, str]]:
        """Get all available persona templates"""
        return self.persona_templates
    
    def get_style_options(self) -> Dict[str, List[str]]:
        """Get all style configuration options"""
        return self.style_options
    
    def rewrite_content(self, chunks: List[Dict[str, Any]], 
                       persona: str,
                       style_settings: Dict[str, str] = None,
                       custom_prompt: str = None) -> List[RewriteResult]:
        """
        Rewrite content chunks in specified persona/style
        
        Args:
            chunks: List of content chunks to rewrite
            persona: Persona key or custom persona description
            style_settings: Style configuration options
            custom_prompt: Custom prompt for advanced users
            
        Returns:
            List of RewriteResult objects
        """
        if style_settings is None:
            style_settings = {
                "tone": "Friendly",
                "complexity": "Moderate", 
                "length": "Same Length",
                "focus": "Keep Original Meaning"
            }
        
        results = []
        
        for chunk in chunks:
            try:
                rewrite_result = self._rewrite_single_chunk(
                    chunk, persona, style_settings, custom_prompt
                )
                results.append(rewrite_result)
                
                # Add small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error rewriting chunk {chunk.get('id', 'unknown')}: {e}")
                # Create fallback result
                fallback_result = self._create_fallback_result(chunk, persona, style_settings)
                results.append(fallback_result)
        
        return results
    
    def _rewrite_single_chunk(self, chunk: Dict[str, Any], 
                             persona: str,
                             style_settings: Dict[str, str],
                             custom_prompt: str = None) -> RewriteResult:
        """Rewrite a single chunk"""
        
        # Create prompt
        if custom_prompt:
            prompt = self._create_custom_prompt(chunk, custom_prompt, style_settings)
        else:
            prompt = self._create_persona_prompt(chunk, persona, style_settings)
        
        try:
            # Import OpenAI here to handle missing dependency gracefully
            import openai
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert storyteller who can adapt any content to different voices, styles, and personas while maintaining the core meaning."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            # Extract rewritten content
            rewritten_text = response.choices[0].message.content.strip()
            
            # Create result object
            result = RewriteResult(
                id=f"{chunk['id']}_rewrite_{persona}",
                original_chunk_id=chunk['id'],
                persona=persona,
                rewritten_text=rewritten_text,
                style_settings=style_settings,
                quality_score=0.9,
                word_count=len(rewritten_text.split())
            )
            
            return result
            
        except ImportError:
            logger.warning("OpenAI not available, using fallback")
            return self._create_fallback_result(chunk, persona, style_settings)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._create_fallback_result(chunk, persona, style_settings)
    
    def _create_persona_prompt(self, chunk: Dict[str, Any], 
                              persona: str,
                              style_settings: Dict[str, str]) -> str:
        """Create prompt for persona-based rewriting"""
        
        text = chunk['text']
        
        # Get persona template
        if persona in self.persona_templates:
            persona_info = self.persona_templates[persona]
            base_prompt = persona_info['prompt_template']
            persona_name = persona_info['name']
        else:
            # Custom persona
            base_prompt = f"Rewrite this story as if told by {persona}. Adapt the voice, style, and perspective to match this persona while keeping the core meaning."
            persona_name = persona
        
        # Add style settings
        style_instructions = self._create_style_instructions(style_settings)
        
        prompt = f"""
{base_prompt}

STYLE SETTINGS:
{style_instructions}

ORIGINAL TEXT:
{text}

INSTRUCTIONS:
- Maintain the core meaning and important information
- Adapt the voice, tone, and style to match {persona_name}
- Apply the specified style settings
- Make it engaging and authentic to the persona
- Keep it natural and readable

REWRITTEN VERSION:
"""
        
        return prompt
    
    def _create_custom_prompt(self, chunk: Dict[str, Any],
                             custom_prompt: str,
                             style_settings: Dict[str, str]) -> str:
        """Create prompt for custom user input"""
        
        text = chunk['text']
        style_instructions = self._create_style_instructions(style_settings)
        
        prompt = f"""
{custom_prompt}

STYLE SETTINGS:
{style_instructions}

ORIGINAL TEXT:
{text}

Apply the custom instructions above to rewrite this text while considering the style settings.

REWRITTEN VERSION:
"""
        
        return prompt
    
    def _create_style_instructions(self, style_settings: Dict[str, str]) -> str:
        """Create style instructions from settings"""
        
        instructions = []
        
        if style_settings.get('tone'):
            instructions.append(f"Tone: {style_settings['tone']}")
        
        if style_settings.get('complexity'):
            instructions.append(f"Language Complexity: {style_settings['complexity']}")
        
        if style_settings.get('length'):
            instructions.append(f"Length: {style_settings['length']}")
        
        if style_settings.get('focus'):
            instructions.append(f"Focus: {style_settings['focus']}")
        
        return "\n".join(instructions)
    
    def _create_fallback_result(self, chunk: Dict[str, Any],
                               persona: str,
                               style_settings: Dict[str, str]) -> RewriteResult:
        """Create fallback result when API fails"""
        
        # Simple fallback transformation
        original_text = chunk['text']
        
        if persona == "3_year_old":
            fallback_text = f"Wow! {original_text} That's so cool!"
        elif persona == "pirate":
            fallback_text = f"Arrr, matey! {original_text} What an adventure!"
        elif persona == "zen_monk":
            fallback_text = f"In mindful contemplation, we observe: {original_text} Such is the way of understanding."
        else:
            fallback_text = f"[{persona} voice]: {original_text}"
        
        return RewriteResult(
            id=f"{chunk['id']}_rewrite_{persona}_fallback",
            original_chunk_id=chunk['id'],
            persona=persona,
            rewritten_text=fallback_text,
            style_settings=style_settings,
            quality_score=0.6,
            word_count=len(fallback_text.split())
        )
    
    def preview_persona(self, persona: str, sample_text: str = None) -> str:
        """Generate a preview of how a persona would rewrite content"""
        
        if sample_text is None:
            sample_text = "The sun was setting over the mountains, painting the sky in brilliant colors."
        
        sample_chunk = {
            'id': 'preview',
            'text': sample_text
        }
        
        result = self._rewrite_single_chunk(
            sample_chunk, 
            persona, 
            {"tone": "Friendly", "complexity": "Moderate", "length": "Same Length", "focus": "Keep Original Meaning"}
        )
        
        return result.rewritten_text
    
    def get_persona_suggestions(self, content_type: str = "general") -> List[str]:
        """Get persona suggestions based on content type"""
        
        suggestions = {
            "children": ["3_year_old", "5_year_old", "friendly_teacher"],
            "educational": ["scientist", "zen_monk", "einstein"],
            "entertainment": ["pirate", "comedian", "shakespeare"],
            "creative": ["poet", "artist", "storyteller"],
            "general": ["3_year_old", "pirate", "zen_monk", "scientist", "comedian"]
        }
        
        return suggestions.get(content_type, suggestions["general"])

