"""
Edit-Refine Loop Engine - Phase 3
=================================

Enables users to iteratively improve generated content through:
- One-click refinement with AI
- Custom refinement prompts
- Quality improvement suggestions
- Version history tracking
- Batch refinement operations
"""

import streamlit as st
import logging
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RefinementRequest:
    """Represents a content refinement request"""
    content_id: str
    original_content: str
    refinement_type: str
    custom_prompt: str = ""
    target_quality: float = 0.9
    preserve_meaning: bool = True

@dataclass
class RefinementResult:
    """Represents the result of a content refinement"""
    request_id: str
    original_content: str
    refined_content: str
    refinement_type: str
    quality_improvement: float
    changes_summary: str
    timestamp: datetime
    version_number: int = 1

class EditRefineEngine:
    """Engine for iterative content improvement"""
    
    def __init__(self):
        self.refinement_types = self._load_refinement_types()
        self.version_history = {}  # Track content versions
        
    def _load_refinement_types(self) -> Dict[str, Dict[str, Any]]:
        """Load different types of content refinements"""
        return {
            "improve_clarity": {
                "name": "Improve Clarity",
                "description": "Make the content clearer and easier to understand",
                "prompt_template": """
Improve the clarity of this content while preserving its original meaning and style.

ORIGINAL CONTENT:
{content}

INSTRUCTIONS:
- Make the language clearer and more precise
- Remove ambiguity and confusion
- Improve sentence structure and flow
- Keep the same tone and style
- Preserve all key information
- Make it easier to understand

IMPROVED CONTENT:
""",
                "focus": "clarity"
            },
            
            "enhance_engagement": {
                "name": "Enhance Engagement",
                "description": "Make the content more engaging and interesting",
                "prompt_template": """
Enhance the engagement of this content while keeping its core message intact.

ORIGINAL CONTENT:
{content}

INSTRUCTIONS:
- Make the content more engaging and interesting
- Add compelling hooks and transitions
- Use more vivid language and examples
- Improve rhythm and pacing
- Keep the original meaning and facts
- Make it more captivating to read

ENHANCED CONTENT:
""",
                "focus": "engagement"
            },
            
            "fix_grammar_style": {
                "name": "Fix Grammar & Style",
                "description": "Correct grammar, punctuation, and improve writing style",
                "prompt_template": """
Fix grammar, punctuation, and improve the writing style of this content.

ORIGINAL CONTENT:
{content}

INSTRUCTIONS:
- Correct all grammar and punctuation errors
- Improve sentence structure and variety
- Enhance word choice and vocabulary
- Ensure consistent style and tone
- Maintain the original meaning
- Make it professionally polished

CORRECTED CONTENT:
""",
                "focus": "grammar"
            },
            
            "adjust_tone": {
                "name": "Adjust Tone",
                "description": "Modify the tone while keeping the content intact",
                "prompt_template": """
Adjust the tone of this content to be more {target_tone} while preserving the information.

ORIGINAL CONTENT:
{content}

TARGET TONE: {target_tone}

INSTRUCTIONS:
- Change the tone to be more {target_tone}
- Keep all the factual information
- Maintain the content structure
- Ensure the new tone is consistent throughout
- Preserve the core message

TONE-ADJUSTED CONTENT:
""",
                "focus": "tone",
                "requires_parameter": "target_tone"
            },
            
            "simplify_language": {
                "name": "Simplify Language",
                "description": "Make the language simpler and more accessible",
                "prompt_template": """
Simplify the language in this content to make it more accessible.

ORIGINAL CONTENT:
{content}

INSTRUCTIONS:
- Use simpler words and shorter sentences
- Replace complex terms with easier alternatives
- Break down complicated ideas
- Keep all important information
- Make it understandable for a broader audience
- Maintain accuracy and completeness

SIMPLIFIED CONTENT:
""",
                "focus": "simplification"
            },
            
            "add_examples": {
                "name": "Add Examples",
                "description": "Add relevant examples and illustrations",
                "prompt_template": """
Enhance this content by adding relevant examples and illustrations.

ORIGINAL CONTENT:
{content}

INSTRUCTIONS:
- Add concrete examples to illustrate key points
- Include relevant analogies or metaphors
- Provide practical applications
- Make abstract concepts more tangible
- Keep the original structure and flow
- Ensure examples are accurate and helpful

ENHANCED WITH EXAMPLES:
""",
                "focus": "examples"
            },
            
            "increase_detail": {
                "name": "Increase Detail",
                "description": "Add more depth and detail to the content",
                "prompt_template": """
Expand this content with more depth and detail while maintaining quality.

ORIGINAL CONTENT:
{content}

INSTRUCTIONS:
- Add relevant details and explanations
- Expand on key concepts
- Provide more context and background
- Include additional insights
- Maintain coherent structure
- Ensure all additions are valuable and relevant

DETAILED CONTENT:
""",
                "focus": "detail"
            },
            
            "custom_refinement": {
                "name": "Custom Refinement",
                "description": "Apply custom refinement instructions",
                "prompt_template": """
Refine this content according to the specific instructions provided.

ORIGINAL CONTENT:
{content}

CUSTOM INSTRUCTIONS:
{custom_instructions}

INSTRUCTIONS:
- Follow the custom instructions carefully
- Preserve the core meaning and information
- Maintain quality and coherence
- Apply the requested changes thoughtfully
- Ensure the result meets the specified requirements

REFINED CONTENT:
""",
                "focus": "custom",
                "requires_parameter": "custom_instructions"
            }
        }
    
    def get_available_refinement_types(self) -> Dict[str, Dict[str, Any]]:
        """Get all available refinement types"""
        return self.refinement_types
    
    def refine_content(self, content: str, content_id: str,
                      refinement_type: str, **kwargs) -> RefinementResult:
        """
        Refine content using specified refinement type
        
        Args:
            content: Original content to refine
            content_id: Unique identifier for the content
            refinement_type: Type of refinement to apply
            **kwargs: Additional parameters for refinement
            
        Returns:
            RefinementResult object
        """
        if refinement_type not in self.refinement_types:
            raise ValueError(f"Unsupported refinement type: {refinement_type}")
        
        try:
            refined_content = self._apply_refinement(
                content, refinement_type, **kwargs
            )
            
            # Calculate quality improvement (simplified metric)
            quality_improvement = self._calculate_quality_improvement(
                content, refined_content, refinement_type
            )
            
            # Generate changes summary
            changes_summary = self._generate_changes_summary(
                content, refined_content, refinement_type
            )
            
            # Create result
            result = RefinementResult(
                request_id=f"{content_id}_{refinement_type}_{int(time.time())}",
                original_content=content,
                refined_content=refined_content,
                refinement_type=refinement_type,
                quality_improvement=quality_improvement,
                changes_summary=changes_summary,
                timestamp=datetime.now(),
                version_number=self._get_next_version_number(content_id)
            )
            
            # Store in version history
            self._store_version(content_id, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error refining content: {e}")
            raise
    
    def _apply_refinement(self, content: str, refinement_type: str, **kwargs) -> str:
        """Apply the specified refinement to content"""
        
        refinement_config = self.refinement_types[refinement_type]
        
        # Prepare prompt
        prompt_template = refinement_config['prompt_template']
        
        # Handle special parameters
        if refinement_type == "adjust_tone":
            target_tone = kwargs.get('target_tone', 'professional')
            prompt = prompt_template.format(
                content=content,
                target_tone=target_tone
            )
        elif refinement_type == "custom_refinement":
            custom_instructions = kwargs.get('custom_instructions', 'Improve the content')
            prompt = prompt_template.format(
                content=content,
                custom_instructions=custom_instructions
            )
        else:
            prompt = prompt_template.format(content=content)
        
        try:
            # Import OpenAI here to handle missing dependency gracefully
            import openai
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert content editor focused on improving text quality while preserving meaning and style."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.3  # Lower temperature for consistent refinement
            )
            
            refined_content = response.choices[0].message.content.strip()
            return refined_content
            
        except ImportError:
            logger.warning("OpenAI not available, using fallback refinement")
            return self._fallback_refinement(content, refinement_type, **kwargs)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._fallback_refinement(content, refinement_type, **kwargs)
    
    def _fallback_refinement(self, content: str, refinement_type: str, **kwargs) -> str:
        """Provide fallback refinement when API is unavailable"""
        
        if refinement_type == "improve_clarity":
            return f"[CLARITY IMPROVED] {content}"
        elif refinement_type == "enhance_engagement":
            return f"[ENGAGEMENT ENHANCED] {content}"
        elif refinement_type == "fix_grammar_style":
            return f"[GRAMMAR FIXED] {content}"
        elif refinement_type == "simplify_language":
            return f"[SIMPLIFIED] {content}"
        elif refinement_type == "add_examples":
            return f"{content}\n\n[EXAMPLE ADDED] For instance, this concept can be applied in real-world scenarios."
        elif refinement_type == "increase_detail":
            return f"{content}\n\n[DETAIL ADDED] This topic involves several important considerations and applications."
        else:
            return f"[REFINED] {content}"
    
    def _calculate_quality_improvement(self, original: str, refined: str, 
                                     refinement_type: str) -> float:
        """Calculate estimated quality improvement"""
        
        # Simple heuristic-based quality improvement calculation
        original_length = len(original.split())
        refined_length = len(refined.split())
        
        # Base improvement based on refinement type
        base_improvements = {
            "improve_clarity": 0.15,
            "enhance_engagement": 0.20,
            "fix_grammar_style": 0.10,
            "adjust_tone": 0.12,
            "simplify_language": 0.18,
            "add_examples": 0.25,
            "increase_detail": 0.30,
            "custom_refinement": 0.15
        }
        
        base_improvement = base_improvements.get(refinement_type, 0.15)
        
        # Adjust based on length change (more content often means more improvement)
        if refinement_type in ["add_examples", "increase_detail"]:
            length_factor = min(refined_length / original_length, 2.0)
            improvement = base_improvement * length_factor
        else:
            improvement = base_improvement
        
        # Cap improvement at reasonable levels
        return min(improvement, 0.5)
    
    def _generate_changes_summary(self, original: str, refined: str, 
                                refinement_type: str) -> str:
        """Generate a summary of changes made"""
        
        original_words = len(original.split())
        refined_words = len(refined.split())
        word_change = refined_words - original_words
        
        refinement_descriptions = {
            "improve_clarity": "Improved clarity and readability",
            "enhance_engagement": "Enhanced engagement and interest",
            "fix_grammar_style": "Fixed grammar and improved style",
            "adjust_tone": "Adjusted tone and voice",
            "simplify_language": "Simplified language and vocabulary",
            "add_examples": "Added examples and illustrations",
            "increase_detail": "Increased depth and detail",
            "custom_refinement": "Applied custom refinement"
        }
        
        base_description = refinement_descriptions.get(
            refinement_type, "Applied refinement"
        )
        
        if word_change > 10:
            return f"{base_description}. Added {word_change} words for better explanation."
        elif word_change < -10:
            return f"{base_description}. Reduced by {abs(word_change)} words for conciseness."
        else:
            return f"{base_description}. Maintained similar length with improved quality."
    
    def _get_next_version_number(self, content_id: str) -> int:
        """Get the next version number for content"""
        if content_id not in self.version_history:
            return 1
        return len(self.version_history[content_id]) + 1
    
    def _store_version(self, content_id: str, result: RefinementResult):
        """Store refinement result in version history"""
        if content_id not in self.version_history:
            self.version_history[content_id] = []
        
        self.version_history[content_id].append(result)
    
    def get_version_history(self, content_id: str) -> List[RefinementResult]:
        """Get version history for content"""
        return self.version_history.get(content_id, [])
    
    def batch_refine(self, content_list: List[Tuple[str, str]], 
                    refinement_type: str, **kwargs) -> List[RefinementResult]:
        """
        Refine multiple content pieces in batch
        
        Args:
            content_list: List of (content_id, content) tuples
            refinement_type: Type of refinement to apply
            **kwargs: Additional parameters for refinement
            
        Returns:
            List of RefinementResult objects
        """
        results = []
        
        for content_id, content in content_list:
            try:
                result = self.refine_content(
                    content, content_id, refinement_type, **kwargs
                )
                results.append(result)
                
                # Add delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error refining content {content_id}: {e}")
                # Create error result
                error_result = RefinementResult(
                    request_id=f"{content_id}_error",
                    original_content=content,
                    refined_content=content,  # Keep original on error
                    refinement_type=refinement_type,
                    quality_improvement=0.0,
                    changes_summary=f"Error during refinement: {str(e)}",
                    timestamp=datetime.now(),
                    version_number=1
                )
                results.append(error_result)
        
        return results
    
    def suggest_refinements(self, content: str, current_quality: float = 0.7) -> List[str]:
        """
        Suggest appropriate refinements based on content analysis
        
        Args:
            content: Content to analyze
            current_quality: Current quality score
            
        Returns:
            List of suggested refinement types
        """
        suggestions = []
        
        # Analyze content characteristics
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Grammar and style suggestions
        if current_quality < 0.8:
            suggestions.append("fix_grammar_style")
        
        # Clarity suggestions
        if avg_sentence_length > 25:
            suggestions.append("improve_clarity")
            suggestions.append("simplify_language")
        
        # Engagement suggestions
        if current_quality < 0.9:
            suggestions.append("enhance_engagement")
        
        # Content depth suggestions
        if word_count < 100:
            suggestions.append("increase_detail")
            suggestions.append("add_examples")
        
        # Always offer custom refinement
        suggestions.append("custom_refinement")
        
        return suggestions[:4]  # Limit to top 4 suggestions
    
    def compare_versions(self, content_id: str, version1: int, version2: int) -> Dict[str, Any]:
        """
        Compare two versions of content
        
        Args:
            content_id: Content identifier
            version1: First version number
            version2: Second version number
            
        Returns:
            Comparison results
        """
        history = self.get_version_history(content_id)
        
        if len(history) < max(version1, version2):
            raise ValueError("Version not found")
        
        v1_result = history[version1 - 1]
        v2_result = history[version2 - 1]
        
        comparison = {
            "version1": {
                "number": version1,
                "content": v1_result.refined_content,
                "quality_improvement": v1_result.quality_improvement,
                "refinement_type": v1_result.refinement_type,
                "timestamp": v1_result.timestamp
            },
            "version2": {
                "number": version2,
                "content": v2_result.refined_content,
                "quality_improvement": v2_result.quality_improvement,
                "refinement_type": v2_result.refinement_type,
                "timestamp": v2_result.timestamp
            },
            "differences": {
                "word_count_change": len(v2_result.refined_content.split()) - len(v1_result.refined_content.split()),
                "quality_change": v2_result.quality_improvement - v1_result.quality_improvement,
                "time_difference": v2_result.timestamp - v1_result.timestamp
            }
        }
        
        return comparison
    
    def export_refinement_history(self, content_id: str, export_format: str = "json") -> str:
        """Export refinement history for content"""
        
        history = self.get_version_history(content_id)
        
        if export_format.lower() == "json":
            export_data = []
            for result in history:
                export_data.append({
                    "version": result.version_number,
                    "refinement_type": result.refinement_type,
                    "content": result.refined_content,
                    "quality_improvement": result.quality_improvement,
                    "changes_summary": result.changes_summary,
                    "timestamp": result.timestamp.isoformat()
                })
            return json.dumps(export_data, indent=2)
        
        elif export_format.lower() == "markdown":
            markdown_content = f"# Refinement History: {content_id}\n\n"
            
            for result in history:
                markdown_content += f"## Version {result.version_number}\n\n"
                markdown_content += f"**Refinement Type:** {result.refinement_type}\n\n"
                markdown_content += f"**Quality Improvement:** +{result.quality_improvement:.1%}\n\n"
                markdown_content += f"**Changes:** {result.changes_summary}\n\n"
                markdown_content += f"**Timestamp:** {result.timestamp}\n\n"
                markdown_content += f"**Content:**\n{result.refined_content}\n\n"
                markdown_content += "---\n\n"
            
            return markdown_content
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def get_refinement_statistics(self, content_id: str = None) -> Dict[str, Any]:
        """Get statistics about refinements"""
        
        if content_id:
            # Statistics for specific content
            history = self.get_version_history(content_id)
            if not history:
                return {"total_refinements": 0}
            
            total_improvement = sum(r.quality_improvement for r in history)
            refinement_types = [r.refinement_type for r in history]
            
            return {
                "total_refinements": len(history),
                "total_quality_improvement": total_improvement,
                "average_improvement": total_improvement / len(history),
                "refinement_types_used": list(set(refinement_types)),
                "latest_version": history[-1].version_number,
                "first_refinement": history[0].timestamp,
                "latest_refinement": history[-1].timestamp
            }
        else:
            # Global statistics
            all_results = []
            for content_history in self.version_history.values():
                all_results.extend(content_history)
            
            if not all_results:
                return {"total_refinements": 0}
            
            refinement_types = [r.refinement_type for r in all_results]
            type_counts = {}
            for rt in refinement_types:
                type_counts[rt] = type_counts.get(rt, 0) + 1
            
            return {
                "total_refinements": len(all_results),
                "total_content_pieces": len(self.version_history),
                "average_refinements_per_content": len(all_results) / len(self.version_history),
                "total_quality_improvement": sum(r.quality_improvement for r in all_results),
                "most_used_refinement_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None,
                "refinement_type_distribution": type_counts
            }

