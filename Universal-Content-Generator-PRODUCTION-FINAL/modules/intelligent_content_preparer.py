"""
Intelligent Content Preparer Module
Uses enhanced spaCy processing to prepare content optimally for GPT transformations
Handles entity preservation, context enhancement, and smart prompt generation
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
from .enhanced_spacy_processor import EnhancedSpacyProcessor

class IntelligentContentPreparer:
    """
    Intelligent content preparation system that uses spaCy analysis
    to optimize content for GPT processing and transformation
    """
    
    def __init__(self):
        """Initialize the intelligent content preparer"""
        self.spacy_processor = EnhancedSpacyProcessor()
        self.logger = logging.getLogger(__name__)
    
    def prepare_content_for_transformation(self, 
                                         text: str, 
                                         transformation_type: str,
                                         style_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Prepare content for GPT transformation using spaCy intelligence
        
        Args:
            text: Input text to prepare
            transformation_type: Type of transformation (rewrite, training_data, etc.)
            style_config: Style configuration parameters
            
        Returns:
            Prepared content with enhanced prompts and context
        """
        if not text or not text.strip():
            return self._empty_preparation()
        
        try:
            # Analyze content with spaCy
            analysis = self.spacy_processor.analyze_content(text)
            
            # Create smart chunks
            chunks = self.spacy_processor.get_smart_chunks(text, max_chunk_size=1000)
            
            # Enhance each chunk for transformation
            enhanced_chunks = []
            for chunk in chunks:
                enhanced_chunk = self._enhance_chunk_for_transformation(
                    chunk, transformation_type, style_config, analysis
                )
                enhanced_chunks.append(enhanced_chunk)
            
            # Create overall preparation summary
            preparation = {
                'original_text': text,
                'transformation_type': transformation_type,
                'style_config': style_config or {},
                'content_analysis': analysis,
                'enhanced_chunks': enhanced_chunks,
                'preparation_metadata': self._create_preparation_metadata(analysis, chunks),
                'recommended_settings': self._recommend_transformation_settings(analysis, transformation_type)
            }
            
            return preparation
            
        except Exception as e:
            self.logger.error(f"Error preparing content: {e}")
            return self._empty_preparation()
    
    def _enhance_chunk_for_transformation(self, 
                                        chunk: Dict[str, Any], 
                                        transformation_type: str,
                                        style_config: Dict[str, Any],
                                        overall_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance individual chunk with transformation-specific context"""
        
        enhanced_chunk = chunk.copy()
        chunk_analysis = chunk.get('analysis', {})
        
        # Extract key information for prompt enhancement
        entities = self._extract_important_entities(chunk_analysis)
        key_phrases = chunk_analysis.get('key_phrases', [])
        context_markers = chunk_analysis.get('context_markers', {})
        
        # Create enhanced prompt based on transformation type
        enhanced_prompt = self._create_enhanced_prompt(
            chunk['text'], transformation_type, style_config, 
            entities, key_phrases, context_markers, overall_analysis
        )
        
        # Add enhancement metadata
        enhanced_chunk.update({
            'enhanced_prompt': enhanced_prompt,
            'important_entities': entities,
            'key_context': self._extract_key_context(chunk_analysis),
            'transformation_hints': self._generate_transformation_hints(
                chunk_analysis, transformation_type
            ),
            'preservation_notes': self._create_preservation_notes(entities, key_phrases)
        })
        
        return enhanced_chunk
    
    def _create_enhanced_prompt(self, 
                              text: str,
                              transformation_type: str,
                              style_config: Dict[str, Any],
                              entities: List[Dict[str, Any]],
                              key_phrases: List[Dict[str, Any]],
                              context_markers: Dict[str, List[str]],
                              overall_analysis: Dict[str, Any]) -> str:
        """Create enhanced prompt with spaCy-derived context"""
        
        base_prompt = self._get_base_prompt(transformation_type, style_config)
        
        # Add entity preservation instructions
        entity_instructions = self._create_entity_instructions(entities)
        
        # Add context preservation instructions
        context_instructions = self._create_context_instructions(context_markers, key_phrases)
        
        # Add style-specific enhancements
        style_enhancements = self._create_style_enhancements(
            style_config, overall_analysis
        )
        
        # Combine all elements
        enhanced_prompt = f"""{base_prompt}

IMPORTANT CONTEXT TO PRESERVE:
{entity_instructions}

{context_instructions}

STYLE GUIDANCE:
{style_enhancements}

CONTENT TO TRANSFORM:
{text}

Please ensure the transformation maintains the important entities and context while achieving the desired style and format."""
        
        return enhanced_prompt
    
    def _get_base_prompt(self, transformation_type: str, style_config: Dict[str, Any]) -> str:
        """Get base prompt template for transformation type"""
        
        prompts = {
            'rewrite_story': self._get_rewrite_prompt(style_config),
            'training_data': self._get_training_data_prompt(style_config),
            'quirky_knowledge': self._get_quirky_knowledge_prompt(style_config),
            'persona_narrator': self._get_persona_narrator_prompt(style_config)
        }
        
        return prompts.get(transformation_type, "Transform the following content:")
    
    def _get_rewrite_prompt(self, style_config: Dict[str, Any]) -> str:
        """Get rewrite transformation prompt"""
        persona = style_config.get('persona', 'general audience')
        tone = style_config.get('tone', 'neutral')
        
        return f"""Rewrite the following content as if written by or for {persona}.
Use a {tone} tone and maintain the core meaning while adapting the style, vocabulary, and perspective appropriately."""
    
    def _get_training_data_prompt(self, style_config: Dict[str, Any]) -> str:
        """Get training data generation prompt"""
        format_type = style_config.get('format', 'qa_pairs')
        
        format_instructions = {
            'qa_pairs': "Create question-answer pairs that capture the key information.",
            'classification': "Create classification examples with appropriate labels.",
            'instruction_following': "Create instruction-response pairs for training.",
            'few_shot': "Create few-shot examples for in-context learning."
        }
        
        instruction = format_instructions.get(format_type, "Create training data examples.")
        
        return f"""Convert the following content into AI training data.
{instruction}
Ensure the examples are clear, accurate, and suitable for machine learning training."""
    
    def _get_quirky_knowledge_prompt(self, style_config: Dict[str, Any]) -> str:
        """Get quirky knowledge transformation prompt"""
        knowledge_type = style_config.get('knowledge_type', 'analogies')
        
        type_instructions = {
            'analogies': "Create interesting analogies that explain the concepts.",
            'metaphors': "Create vivid metaphors that illuminate the ideas.",
            'socratic_qa': "Create Socratic-style questions that lead to understanding.",
            'riddles': "Create thought-provoking riddles based on the content."
        }
        
        instruction = type_instructions.get(knowledge_type, "Create engaging knowledge tools.")
        
        return f"""Transform the following content into quirky knowledge tools.
{instruction}
Make the output engaging, memorable, and educational."""
    
    def _get_persona_narrator_prompt(self, style_config: Dict[str, Any]) -> str:
        """Get persona narrator transformation prompt"""
        persona = style_config.get('persona', 'wise teacher')
        
        return f"""Rewrite the following content as if narrated by {persona}.
Adopt their characteristic voice, perspective, vocabulary, and way of explaining concepts.
Maintain the factual accuracy while infusing the content with the persona's unique style."""
    
    def _extract_important_entities(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and prioritize important entities"""
        entities = analysis.get('entities', {})
        important_entities = []
        
        # Priority order for entity types
        priority_types = ['PERSON', 'ORG', 'GPE', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE']
        
        for entity_type in priority_types:
            if entity_type in entities:
                for entity in entities[entity_type]:
                    important_entities.append({
                        'text': entity['text'],
                        'type': entity_type,
                        'description': entity.get('description', ''),
                        'importance': 'high' if entity_type in ['PERSON', 'ORG'] else 'medium'
                    })
        
        # Add other entities with lower priority
        for entity_type, entity_list in entities.items():
            if entity_type not in priority_types:
                for entity in entity_list:
                    important_entities.append({
                        'text': entity['text'],
                        'type': entity_type,
                        'description': entity.get('description', ''),
                        'importance': 'low'
                    })
        
        return important_entities[:10]  # Top 10 most important
    
    def _create_entity_instructions(self, entities: List[Dict[str, Any]]) -> str:
        """Create instructions for preserving important entities"""
        if not entities:
            return "No specific entities to preserve."
        
        high_priority = [e for e in entities if e['importance'] == 'high']
        medium_priority = [e for e in entities if e['importance'] == 'medium']
        
        instructions = []
        
        if high_priority:
            entity_names = [e['text'] for e in high_priority]
            instructions.append(f"MUST preserve these important names/organizations: {', '.join(entity_names)}")
        
        if medium_priority:
            entity_names = [e['text'] for e in medium_priority]
            instructions.append(f"Should preserve these entities when possible: {', '.join(entity_names)}")
        
        return '\n'.join(instructions) if instructions else "Preserve key entities mentioned in the text."
    
    def _extract_key_context(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key contextual information"""
        return {
            'complexity_level': analysis.get('complexity', {}).get('overall_score', 0),
            'sentiment': analysis.get('sentiment', {}).get('polarity', 0),
            'main_topics': [topic['phrase'] for topic in analysis.get('topics', [])[:5]],
            'structure_type': self._determine_structure_type(analysis.get('structure', {})),
            'discourse_style': self._determine_discourse_style(analysis.get('context_markers', {}))
        }
    
    def _create_context_instructions(self, 
                                   context_markers: Dict[str, List[str]], 
                                   key_phrases: List[Dict[str, Any]]) -> str:
        """Create instructions for preserving context"""
        instructions = []
        
        # Temporal context
        if context_markers.get('temporal'):
            instructions.append("Preserve temporal relationships and sequence.")
        
        # Causal context
        if context_markers.get('causal'):
            instructions.append("Maintain cause-and-effect relationships.")
        
        # Key phrases
        if key_phrases:
            top_phrases = [phrase['text'] for phrase in key_phrases[:3]]
            instructions.append(f"Preserve these key concepts: {', '.join(top_phrases)}")
        
        return '\n'.join(instructions) if instructions else "Maintain the logical flow and key concepts."
    
    def _create_style_enhancements(self, 
                                 style_config: Dict[str, Any], 
                                 analysis: Dict[str, Any]) -> str:
        """Create style-specific enhancement instructions"""
        enhancements = []
        
        # Complexity adaptation
        complexity = analysis.get('complexity', {}).get('overall_score', 0)
        if complexity > 0.7:
            enhancements.append("The original content is complex - adapt appropriately for the target audience.")
        elif complexity < 0.3:
            enhancements.append("The original content is simple - maintain clarity while adding appropriate depth.")
        
        # Tone guidance
        tone = style_config.get('tone', 'neutral')
        if tone != 'neutral':
            enhancements.append(f"Adopt a {tone} tone throughout the transformation.")
        
        # Length guidance
        length = style_config.get('length', 'medium')
        length_instructions = {
            'short': "Keep the output concise and focused.",
            'medium': "Provide balanced detail and explanation.",
            'detailed': "Include comprehensive explanations and examples."
        }
        if length in length_instructions:
            enhancements.append(length_instructions[length])
        
        return '\n'.join(enhancements) if enhancements else "Maintain appropriate style and tone."
    
    def _generate_transformation_hints(self, 
                                     analysis: Dict[str, Any], 
                                     transformation_type: str) -> List[str]:
        """Generate transformation-specific hints based on content analysis"""
        hints = []
        
        structure = analysis.get('structure', {})
        
        # Hints based on content structure
        if structure.get('question_answer_pairs'):
            hints.append("Content contains Q&A patterns - consider preserving this structure.")
        
        if structure.get('dialogue_indicators', {}).get('quotation_marks', 0) > 0:
            hints.append("Content contains dialogue - handle quoted speech appropriately.")
        
        if structure.get('list_structures', {}).get('numbered_lists'):
            hints.append("Content has numbered lists - consider maintaining organization.")
        
        # Hints based on transformation type
        if transformation_type == 'training_data':
            if analysis.get('topics'):
                hints.append("Rich topic content - good for diverse training examples.")
        
        elif transformation_type == 'rewrite_story':
            sentiment = analysis.get('sentiment', {}).get('polarity', 0)
            if abs(sentiment) > 0.3:
                hints.append(f"Content has {'positive' if sentiment > 0 else 'negative'} sentiment - consider in rewrite.")
        
        return hints
    
    def _create_preservation_notes(self, 
                                 entities: List[Dict[str, Any]], 
                                 key_phrases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create notes about what should be preserved"""
        return {
            'critical_entities': [e['text'] for e in entities if e['importance'] == 'high'],
            'important_phrases': [p['text'] for p in key_phrases[:5]],
            'preservation_priority': 'high' if entities else 'medium'
        }
    
    def _determine_structure_type(self, structure: Dict[str, Any]) -> str:
        """Determine the primary structure type of the content"""
        if structure.get('question_answer_pairs'):
            return 'qa_format'
        elif structure.get('dialogue_indicators', {}).get('quotation_marks', 0) > 2:
            return 'dialogue'
        elif structure.get('list_structures', {}).get('numbered_lists'):
            return 'structured_list'
        elif structure.get('sentence_types', {}).get('declarative', 0) > 0.8:
            return 'expository'
        else:
            return 'mixed'
    
    def _determine_discourse_style(self, context_markers: Dict[str, List[str]]) -> str:
        """Determine the discourse style based on context markers"""
        if context_markers.get('causal'):
            return 'analytical'
        elif context_markers.get('temporal'):
            return 'narrative'
        elif context_markers.get('comparative'):
            return 'argumentative'
        elif context_markers.get('emphasis'):
            return 'persuasive'
        else:
            return 'descriptive'
    
    def _create_preparation_metadata(self, 
                                   analysis: Dict[str, Any], 
                                   chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create metadata about the preparation process"""
        return {
            'total_chunks': len(chunks),
            'avg_chunk_size': sum(len(chunk['text']) for chunk in chunks) / len(chunks) if chunks else 0,
            'content_complexity': analysis.get('complexity', {}).get('overall_score', 0),
            'entity_count': sum(len(entities) for entities in analysis.get('entities', {}).values()),
            'key_phrase_count': len(analysis.get('key_phrases', [])),
            'preparation_quality': self._assess_preparation_quality(analysis, chunks)
        }
    
    def _recommend_transformation_settings(self, 
                                         analysis: Dict[str, Any], 
                                         transformation_type: str) -> Dict[str, Any]:
        """Recommend optimal settings based on content analysis"""
        recommendations = {}
        
        complexity = analysis.get('complexity', {}).get('overall_score', 0)
        
        # Complexity-based recommendations
        if complexity > 0.7:
            recommendations['suggested_length'] = 'detailed'
            recommendations['complexity_note'] = 'Complex content - may need detailed explanation'
        elif complexity < 0.3:
            recommendations['suggested_length'] = 'short'
            recommendations['complexity_note'] = 'Simple content - can be concise'
        else:
            recommendations['suggested_length'] = 'medium'
        
        # Sentiment-based recommendations
        sentiment = analysis.get('sentiment', {}).get('polarity', 0)
        if abs(sentiment) > 0.3:
            recommendations['tone_note'] = f"Content has {'positive' if sentiment > 0 else 'negative'} sentiment"
        
        # Structure-based recommendations
        structure = analysis.get('structure', {})
        if structure.get('question_answer_pairs'):
            recommendations['format_suggestion'] = 'Consider Q&A format preservation'
        
        return recommendations
    
    def _assess_preparation_quality(self, 
                                  analysis: Dict[str, Any], 
                                  chunks: List[Dict[str, Any]]) -> str:
        """Assess the quality of content preparation"""
        score = 0
        
        # Entity richness
        entity_count = sum(len(entities) for entities in analysis.get('entities', {}).values())
        if entity_count > 5:
            score += 0.3
        elif entity_count > 0:
            score += 0.1
        
        # Key phrase richness
        key_phrases = len(analysis.get('key_phrases', []))
        if key_phrases > 5:
            score += 0.3
        elif key_phrases > 0:
            score += 0.1
        
        # Chunk quality
        if chunks:
            avg_chunk_analysis_quality = sum(
                1 for chunk in chunks 
                if chunk.get('analysis', {}).get('entities')
            ) / len(chunks)
            score += avg_chunk_analysis_quality * 0.4
        
        if score > 0.7:
            return 'excellent'
        elif score > 0.4:
            return 'good'
        elif score > 0.2:
            return 'fair'
        else:
            return 'basic'
    
    def _empty_preparation(self) -> Dict[str, Any]:
        """Return empty preparation structure"""
        return {
            'original_text': '',
            'transformation_type': '',
            'style_config': {},
            'content_analysis': {},
            'enhanced_chunks': [],
            'preparation_metadata': {},
            'recommended_settings': {}
        }

