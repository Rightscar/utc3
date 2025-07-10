"""
Quirky Knowledge Tools Generator - Core Use Case 3
=================================================

Converts any content into creative knowledge formats:
- Analogies and metaphors
- Socratic questioning
- Riddles and puzzles
- Mind maps and concept connections
- Creative explanations

Features:
- Multiple knowledge transformation types
- Educational and entertaining outputs
- Customizable complexity levels
- Interactive learning formats
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
class QuirkyKnowledgeItem:
    """Represents a quirky knowledge transformation result"""
    id: str
    original_chunk_id: str
    knowledge_type: str
    title: str
    content: str
    explanation: str
    difficulty_level: str
    educational_value: float = 0.8
    entertainment_value: float = 0.7

class QuirkyKnowledgeGenerator:
    """Generate creative knowledge transformations from content"""
    
    def __init__(self):
        self.knowledge_types = self._load_knowledge_types()
        self.difficulty_levels = ["Beginner", "Intermediate", "Advanced", "Expert"]
        
    def _load_knowledge_types(self) -> Dict[str, Dict[str, Any]]:
        """Load different types of quirky knowledge transformations"""
        return {
            "analogies": {
                "name": "Analogies & Metaphors",
                "description": "Transform concepts into relatable analogies",
                "prompt_template": """
Create creative analogies and metaphors to explain the concepts in this content.

CONTENT:
{content}

REQUIREMENTS:
- Generate {num_items} creative analogies/metaphors
- Make complex concepts relatable and memorable
- Use everyday objects and experiences
- Include both simple and sophisticated comparisons
- Explain why each analogy works

DIFFICULTY: {difficulty}

FORMAT: Return as JSON array:
[
  {{
    "concept": "The main concept being explained",
    "analogy": "Creative analogy or metaphor",
    "explanation": "Why this analogy works and what it teaches",
    "everyday_example": "Concrete example people can relate to",
    "learning_insight": "Key insight this analogy reveals"
  }}
]
""",
                "examples": [
                    "Meditation is like tuning a radio - you adjust your attention until you find the clear signal of awareness",
                    "Memory is like a library where books are constantly being reshuffled by the librarian of experience"
                ]
            },
            
            "socratic_questions": {
                "name": "Socratic Questioning",
                "description": "Generate thought-provoking questions that lead to deeper understanding",
                "prompt_template": """
Create Socratic questions that guide learners to discover insights from this content.

CONTENT:
{content}

REQUIREMENTS:
- Generate {num_items} thought-provoking questions
- Questions should lead to self-discovery
- Build from simple to complex understanding
- Encourage critical thinking
- Include follow-up questions

DIFFICULTY: {difficulty}

FORMAT: Return as JSON array:
[
  {{
    "main_question": "Primary Socratic question",
    "follow_up_questions": ["Question 1", "Question 2", "Question 3"],
    "thinking_direction": "What this question helps explore",
    "deeper_insight": "The deeper understanding this leads to",
    "real_world_application": "How to apply this thinking"
  }}
]
""",
                "examples": [
                    "If awareness is always present, what is it that seems to come and go?",
                    "What would change in your life if you truly believed your thoughts were just mental events?"
                ]
            },
            
            "riddles_puzzles": {
                "name": "Riddles & Puzzles",
                "description": "Transform knowledge into engaging riddles and puzzles",
                "prompt_template": """
Create educational riddles and puzzles based on this content.

CONTENT:
{content}

REQUIREMENTS:
- Generate {num_items} clever riddles/puzzles
- Make learning fun and memorable
- Include different puzzle types (word play, logic, visual)
- Provide clear solutions with explanations
- Connect back to the original concepts

DIFFICULTY: {difficulty}

FORMAT: Return as JSON array:
[
  {{
    "riddle_or_puzzle": "The riddle or puzzle question",
    "puzzle_type": "word_play/logic/visual/conceptual",
    "solution": "The answer or solution",
    "explanation": "How the solution connects to the concept",
    "learning_objective": "What this puzzle teaches",
    "hint": "Optional hint for learners"
  }}
]
""",
                "examples": [
                    "I am always with you but never seen, always present but never grasped. What am I? (Answer: Awareness)",
                    "The more you chase me, the more I run away. The more you ignore me, the closer I come. What am I? (Answer: Peace of mind)"
                ]
            },
            
            "concept_connections": {
                "name": "Concept Connections",
                "description": "Reveal surprising connections between ideas",
                "prompt_template": """
Discover and explain surprising connections between concepts in this content.

CONTENT:
{content}

REQUIREMENTS:
- Generate {num_items} unexpected concept connections
- Show how seemingly unrelated ideas connect
- Reveal hidden patterns and relationships
- Make connections memorable and insightful
- Include practical implications

DIFFICULTY: {difficulty}

FORMAT: Return as JSON array:
[
  {{
    "concept_a": "First concept",
    "concept_b": "Second concept", 
    "connection_type": "Type of connection (causal/analogical/structural/etc)",
    "connection_explanation": "How these concepts connect",
    "surprising_insight": "The unexpected insight this reveals",
    "practical_application": "How to use this connection"
  }}
]
""",
                "examples": [
                    "Breathing and attention both have rhythm - when one becomes irregular, the other follows",
                    "Learning and forgetting are both forms of change - mastery comes from changing in the right direction"
                ]
            },
            
            "creative_explanations": {
                "name": "Creative Explanations",
                "description": "Explain concepts through stories, scenarios, and creative formats",
                "prompt_template": """
Create creative explanations for the concepts in this content using stories, scenarios, or unique formats.

CONTENT:
{content}

REQUIREMENTS:
- Generate {num_items} creative explanations
- Use storytelling, scenarios, or unique formats
- Make abstract concepts concrete and memorable
- Include characters, settings, or narrative elements
- Ensure explanations are both accurate and engaging

DIFFICULTY: {difficulty}

FORMAT: Return as JSON array:
[
  {{
    "concept": "The concept being explained",
    "creative_format": "story/scenario/dialogue/journey/etc",
    "explanation": "The creative explanation",
    "key_characters": ["Character 1", "Character 2"],
    "main_lesson": "The key lesson or insight",
    "memorable_moment": "The most memorable part of the explanation"
  }}
]
""",
                "examples": [
                    "Meditation as a journey where thoughts are like clouds passing through the sky of awareness",
                    "Learning as building a house where each new concept is a brick that must fit with all the others"
                ]
            },
            
            "paradox_exploration": {
                "name": "Paradox Exploration",
                "description": "Explore paradoxes and contradictions that lead to deeper understanding",
                "prompt_template": """
Identify and explore paradoxes within this content that lead to deeper understanding.

CONTENT:
{content}

REQUIREMENTS:
- Generate {num_items} thought-provoking paradoxes
- Show how contradictions can reveal truth
- Explore the tension between opposing ideas
- Lead to deeper understanding through paradox
- Include resolution or acceptance of the paradox

DIFFICULTY: {difficulty}

FORMAT: Return as JSON array:
[
  {{
    "paradox_statement": "The paradoxical statement or situation",
    "contradiction": "What seems contradictory",
    "deeper_truth": "The deeper truth the paradox reveals",
    "resolution_approach": "How to work with this paradox",
    "wisdom_insight": "The wisdom that emerges from embracing the paradox",
    "practical_example": "A practical example of this paradox"
  }}
]
""",
                "examples": [
                    "The more you try to control your thoughts, the more out of control they become",
                    "True knowledge begins with knowing that you don't know"
                ]
            }
        }
    
    def get_available_knowledge_types(self) -> Dict[str, Dict[str, Any]]:
        """Get all available knowledge transformation types"""
        return self.knowledge_types
    
    def get_difficulty_levels(self) -> List[str]:
        """Get available difficulty levels"""
        return self.difficulty_levels
    
    def generate_quirky_knowledge(self, chunks: List[Dict[str, Any]],
                                 knowledge_type: str,
                                 difficulty: str = "Intermediate",
                                 num_items_per_chunk: int = 3) -> List[QuirkyKnowledgeItem]:
        """
        Generate quirky knowledge transformations from content chunks
        
        Args:
            chunks: List of content chunks
            knowledge_type: Type of knowledge transformation
            difficulty: Difficulty level
            num_items_per_chunk: Number of items to generate per chunk
            
        Returns:
            List of QuirkyKnowledgeItem objects
        """
        if knowledge_type not in self.knowledge_types:
            raise ValueError(f"Unsupported knowledge type: {knowledge_type}")
        
        results = []
        
        for chunk in chunks:
            try:
                chunk_results = self._generate_chunk_quirky_knowledge(
                    chunk, knowledge_type, difficulty, num_items_per_chunk
                )
                results.extend(chunk_results)
                
                # Add small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error generating quirky knowledge for chunk {chunk.get('id', 'unknown')}: {e}")
                # Create fallback knowledge
                fallback_items = self._create_fallback_quirky_knowledge(
                    chunk, knowledge_type, difficulty
                )
                results.extend(fallback_items)
        
        return results
    
    def _generate_chunk_quirky_knowledge(self, chunk: Dict[str, Any],
                                       knowledge_type: str,
                                       difficulty: str,
                                       num_items: int) -> List[QuirkyKnowledgeItem]:
        """Generate quirky knowledge for a single chunk"""
        
        template = self.knowledge_types[knowledge_type]
        prompt = template['prompt_template'].format(
            content=chunk['text'],
            num_items=num_items,
            difficulty=difficulty
        )
        
        try:
            # Import OpenAI here to handle missing dependency gracefully
            import openai
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative educator who transforms knowledge into engaging, memorable formats. Always return valid JSON arrays as specified."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.8  # Higher temperature for creativity
            )
            
            # Parse response
            content = response.choices[0].message.content
            knowledge_items = self._parse_quirky_knowledge_response(
                content, chunk['id'], knowledge_type, difficulty
            )
            
            return knowledge_items
            
        except ImportError:
            logger.warning("OpenAI not available, using fallback")
            return self._create_fallback_quirky_knowledge(chunk, knowledge_type, difficulty)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._create_fallback_quirky_knowledge(chunk, knowledge_type, difficulty)
    
    def _parse_quirky_knowledge_response(self, content: str, chunk_id: str,
                                       knowledge_type: str, difficulty: str) -> List[QuirkyKnowledgeItem]:
        """Parse GPT response into QuirkyKnowledgeItem objects"""
        
        knowledge_items = []
        
        try:
            # Try to parse as JSON
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_content = content[json_start:json_end]
                parsed_data = json.loads(json_content)
                
                for i, item in enumerate(parsed_data):
                    # Extract title and content based on knowledge type
                    if knowledge_type == "analogies":
                        title = item.get('concept', f'Analogy {i+1}')
                        content_text = item.get('analogy', '')
                        explanation = item.get('explanation', '')
                    elif knowledge_type == "socratic_questions":
                        title = "Socratic Question"
                        content_text = item.get('main_question', '')
                        explanation = item.get('thinking_direction', '')
                    elif knowledge_type == "riddles_puzzles":
                        title = f"{item.get('puzzle_type', 'Puzzle').title()} Puzzle"
                        content_text = item.get('riddle_or_puzzle', '')
                        explanation = item.get('explanation', '')
                    elif knowledge_type == "concept_connections":
                        title = f"Connection: {item.get('concept_a', '')} ↔ {item.get('concept_b', '')}"
                        content_text = item.get('connection_explanation', '')
                        explanation = item.get('surprising_insight', '')
                    elif knowledge_type == "creative_explanations":
                        title = f"Creative Explanation: {item.get('concept', '')}"
                        content_text = item.get('explanation', '')
                        explanation = item.get('main_lesson', '')
                    elif knowledge_type == "paradox_exploration":
                        title = "Paradox"
                        content_text = item.get('paradox_statement', '')
                        explanation = item.get('deeper_truth', '')
                    else:
                        title = f"Knowledge Item {i+1}"
                        content_text = str(item)
                        explanation = "Generated knowledge transformation"
                    
                    knowledge_item = QuirkyKnowledgeItem(
                        id=f"{chunk_id}_{knowledge_type}_{i}",
                        original_chunk_id=chunk_id,
                        knowledge_type=knowledge_type,
                        title=title,
                        content=content_text,
                        explanation=explanation,
                        difficulty_level=difficulty,
                        educational_value=0.9,
                        entertainment_value=0.8
                    )
                    knowledge_items.append(knowledge_item)
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from GPT response for chunk {chunk_id}")
            # Create fallback items
            knowledge_items = self._create_simple_fallback_knowledge(
                content, chunk_id, knowledge_type, difficulty
            )
        
        return knowledge_items
    
    def _create_fallback_quirky_knowledge(self, chunk: Dict[str, Any],
                                        knowledge_type: str,
                                        difficulty: str) -> List[QuirkyKnowledgeItem]:
        """Create fallback quirky knowledge when API fails"""
        
        text = chunk['text']
        chunk_id = chunk['id']
        
        fallback_items = []
        
        if knowledge_type == "analogies":
            item = QuirkyKnowledgeItem(
                id=f"{chunk_id}_analogy_fallback",
                original_chunk_id=chunk_id,
                knowledge_type="analogies",
                title="Simple Analogy",
                content=f"This concept is like a puzzle - each piece of understanding fits together to create the complete picture.",
                explanation="This analogy helps visualize how knowledge builds upon itself.",
                difficulty_level=difficulty,
                educational_value=0.6,
                entertainment_value=0.5
            )
            fallback_items.append(item)
        
        elif knowledge_type == "socratic_questions":
            item = QuirkyKnowledgeItem(
                id=f"{chunk_id}_socratic_fallback",
                original_chunk_id=chunk_id,
                knowledge_type="socratic_questions",
                title="Reflective Question",
                content="What would happen if you applied this knowledge in your daily life?",
                explanation="This question encourages practical application and deeper reflection.",
                difficulty_level=difficulty,
                educational_value=0.7,
                entertainment_value=0.6
            )
            fallback_items.append(item)
        
        elif knowledge_type == "riddles_puzzles":
            item = QuirkyKnowledgeItem(
                id=f"{chunk_id}_riddle_fallback",
                original_chunk_id=chunk_id,
                knowledge_type="riddles_puzzles",
                title="Knowledge Riddle",
                content="I am learned but not taught, understood but not explained. What am I?",
                explanation="Answer: Wisdom - it comes through experience and insight rather than instruction.",
                difficulty_level=difficulty,
                educational_value=0.6,
                entertainment_value=0.7
            )
            fallback_items.append(item)
        
        else:
            # Generic fallback
            item = QuirkyKnowledgeItem(
                id=f"{chunk_id}_{knowledge_type}_fallback",
                original_chunk_id=chunk_id,
                knowledge_type=knowledge_type,
                title=f"Knowledge Insight",
                content=f"This content contains valuable insights that can be explored through {knowledge_type.replace('_', ' ')}.",
                explanation="A creative approach to understanding this material.",
                difficulty_level=difficulty,
                educational_value=0.5,
                entertainment_value=0.5
            )
            fallback_items.append(item)
        
        return fallback_items
    
    def _create_simple_fallback_knowledge(self, content: str, chunk_id: str,
                                        knowledge_type: str, difficulty: str) -> List[QuirkyKnowledgeItem]:
        """Create simple fallback knowledge from unparseable content"""
        
        # Extract any useful content
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        items = []
        for i, line in enumerate(lines[:2]):  # Max 2 items
            item = QuirkyKnowledgeItem(
                id=f"{chunk_id}_{knowledge_type}_simple_{i}",
                original_chunk_id=chunk_id,
                knowledge_type=knowledge_type,
                title=f"Knowledge Item {i+1}",
                content=line[:300],
                explanation="Generated from content analysis",
                difficulty_level=difficulty,
                educational_value=0.5,
                entertainment_value=0.4
            )
            items.append(item)
        
        return items
    
    def generate_mixed_knowledge_set(self, chunks: List[Dict[str, Any]],
                                   difficulty: str = "Intermediate") -> Dict[str, List[QuirkyKnowledgeItem]]:
        """Generate a mixed set of all knowledge types from chunks"""
        
        mixed_results = {}
        
        for knowledge_type in self.knowledge_types.keys():
            try:
                results = self.generate_quirky_knowledge(
                    chunks, knowledge_type, difficulty, num_items_per_chunk=2
                )
                mixed_results[knowledge_type] = results
            except Exception as e:
                logger.error(f"Error generating {knowledge_type}: {e}")
                mixed_results[knowledge_type] = []
        
        return mixed_results
    
    def export_quirky_knowledge(self, knowledge_items: List[QuirkyKnowledgeItem],
                               export_format: str = "json") -> str:
        """Export quirky knowledge in specified format"""
        
        if export_format.lower() == "json":
            export_data = []
            for item in knowledge_items:
                export_data.append({
                    "id": item.id,
                    "type": item.knowledge_type,
                    "title": item.title,
                    "content": item.content,
                    "explanation": item.explanation,
                    "difficulty": item.difficulty_level,
                    "educational_value": item.educational_value,
                    "entertainment_value": item.entertainment_value
                })
            return json.dumps(export_data, indent=2)
        
        elif export_format.lower() == "markdown":
            markdown_content = "# Quirky Knowledge Collection\n\n"
            
            current_type = None
            for item in knowledge_items:
                if item.knowledge_type != current_type:
                    current_type = item.knowledge_type
                    type_name = self.knowledge_types[current_type]["name"]
                    markdown_content += f"## {type_name}\n\n"
                
                markdown_content += f"### {item.title}\n\n"
                markdown_content += f"**Content:** {item.content}\n\n"
                markdown_content += f"**Explanation:** {item.explanation}\n\n"
                markdown_content += f"**Difficulty:** {item.difficulty_level}\n\n"
                markdown_content += "---\n\n"
            
            return markdown_content
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def get_knowledge_statistics(self, knowledge_items: List[QuirkyKnowledgeItem]) -> Dict[str, Any]:
        """Get statistics about generated quirky knowledge"""
        
        if not knowledge_items:
            return {"total_items": 0}
        
        stats = {
            "total_items": len(knowledge_items),
            "types_distribution": {},
            "difficulty_distribution": {},
            "average_educational_value": sum(item.educational_value for item in knowledge_items) / len(knowledge_items),
            "average_entertainment_value": sum(item.entertainment_value for item in knowledge_items) / len(knowledge_items),
            "quality_score": 0
        }
        
        # Calculate distributions
        for item in knowledge_items:
            # Type distribution
            type_name = self.knowledge_types[item.knowledge_type]["name"]
            stats["types_distribution"][type_name] = stats["types_distribution"].get(type_name, 0) + 1
            
            # Difficulty distribution
            stats["difficulty_distribution"][item.difficulty_level] = stats["difficulty_distribution"].get(item.difficulty_level, 0) + 1
        
        # Calculate overall quality score
        stats["quality_score"] = (stats["average_educational_value"] + stats["average_entertainment_value"]) / 2
        
        return stats

