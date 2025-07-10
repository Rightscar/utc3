"""
AI Training Data Generator - Core Use Case 2
===========================================

Converts any content into structured training data formats.
Ideal for fine-tuning, few-shot learning, and AI model training.

Supported Formats:
- JSON (structured data)
- JSONL (line-delimited JSON)
- Q&A pairs (question-answer format)
- Classification data (text-label pairs)
- Instruction-following data
- Few-shot examples
"""

import streamlit as st
import logging
import json
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingDataItem:
    """Represents a single training data item"""
    id: str
    format_type: str
    input_text: str
    output_text: str
    metadata: Dict[str, Any]
    quality_score: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def to_jsonl(self) -> str:
        """Convert to JSONL format (single line)"""
        return json.dumps(self.to_dict())

class AITrainingDataGenerator:
    """Generate AI training data from content chunks"""
    
    def __init__(self):
        self.format_templates = self._load_format_templates()
        self.data_types = self._load_data_types()
        
    def _load_format_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load training data format templates"""
        return {
            "qa_pairs": {
                "name": "Question-Answer Pairs",
                "description": "Generate Q&A pairs for training conversational AI",
                "schema": {
                    "question": "str",
                    "answer": "str", 
                    "context": "str (optional)",
                    "difficulty": "str (easy/medium/hard)"
                },
                "use_cases": ["Chatbots", "FAQ systems", "Educational AI"],
                "prompt_template": """
Create high-quality question-answer pairs from this content for AI training.

CONTENT:
{content}

REQUIREMENTS:
- Generate {num_items} diverse Q&A pairs
- Questions should be clear and specific
- Answers should be comprehensive but concise
- Include different difficulty levels
- Ensure answers are grounded in the content

FORMAT: Return as JSON array:
[
  {{
    "question": "Clear, specific question",
    "answer": "Comprehensive answer based on content",
    "context": "Relevant context from original text",
    "difficulty": "easy/medium/hard",
    "topics": ["topic1", "topic2"]
  }}
]
"""
            },
            
            "classification": {
                "name": "Text Classification Data",
                "description": "Generate text-label pairs for classification training",
                "schema": {
                    "text": "str",
                    "label": "str",
                    "confidence": "float",
                    "category": "str"
                },
                "use_cases": ["Sentiment analysis", "Topic classification", "Content moderation"],
                "prompt_template": """
Create text classification training data from this content.

CONTENT:
{content}

CLASSIFICATION TASK: {classification_task}
POSSIBLE LABELS: {labels}

REQUIREMENTS:
- Generate {num_items} text-label pairs
- Extract diverse text segments
- Assign appropriate labels
- Include confidence scores
- Ensure balanced representation

FORMAT: Return as JSON array:
[
  {{
    "text": "Text segment for classification",
    "label": "assigned_label",
    "confidence": 0.95,
    "category": "content_category",
    "reasoning": "Why this label was chosen"
  }}
]
"""
            },
            
            "instruction_following": {
                "name": "Instruction-Following Data",
                "description": "Generate instruction-response pairs for training",
                "schema": {
                    "instruction": "str",
                    "input": "str (optional)",
                    "output": "str",
                    "task_type": "str"
                },
                "use_cases": ["General AI training", "Task-specific models", "Assistant training"],
                "prompt_template": """
Create instruction-following training data from this content.

CONTENT:
{content}

REQUIREMENTS:
- Generate {num_items} instruction-response pairs
- Instructions should be clear and actionable
- Responses should follow instructions precisely
- Include various task types (summarize, explain, analyze, etc.)
- Ensure high-quality examples

FORMAT: Return as JSON array:
[
  {{
    "instruction": "Clear instruction for the AI",
    "input": "Input text (if applicable)",
    "output": "Expected response following the instruction",
    "task_type": "summarization/explanation/analysis/etc"
  }}
]
"""
            },
            
            "few_shot_examples": {
                "name": "Few-Shot Learning Examples",
                "description": "Generate examples for few-shot prompting",
                "schema": {
                    "input": "str",
                    "output": "str",
                    "task_description": "str",
                    "example_type": "str"
                },
                "use_cases": ["Prompt engineering", "Few-shot learning", "In-context learning"],
                "prompt_template": """
Create few-shot learning examples from this content.

CONTENT:
{content}

TASK: {task_description}

REQUIREMENTS:
- Generate {num_items} high-quality input-output examples
- Examples should demonstrate the task clearly
- Include diverse scenarios and edge cases
- Ensure consistency in format and quality
- Make examples suitable for few-shot prompting

FORMAT: Return as JSON array:
[
  {{
    "input": "Example input for the task",
    "output": "Expected output for this input",
    "task_description": "What this example demonstrates",
    "example_type": "basic/advanced/edge_case"
  }}
]
"""
            },
            
            "dialogue_data": {
                "name": "Dialogue Training Data",
                "description": "Generate conversational data for dialogue systems",
                "schema": {
                    "user_message": "str",
                    "assistant_response": "str",
                    "context": "str",
                    "intent": "str"
                },
                "use_cases": ["Chatbots", "Virtual assistants", "Conversational AI"],
                "prompt_template": """
Create dialogue training data from this content.

CONTENT:
{content}

REQUIREMENTS:
- Generate {num_items} realistic dialogue exchanges
- User messages should be natural and varied
- Assistant responses should be helpful and accurate
- Include different conversation types and intents
- Ensure responses are grounded in the content

FORMAT: Return as JSON array:
[
  {{
    "user_message": "Natural user question or statement",
    "assistant_response": "Helpful, accurate response",
    "context": "Relevant background context",
    "intent": "user_intent_category",
    "topics": ["topic1", "topic2"]
  }}
]
"""
            },
            
            "summarization_data": {
                "name": "Summarization Training Data",
                "description": "Generate text-summary pairs for summarization models",
                "schema": {
                    "source_text": "str",
                    "summary": "str",
                    "summary_type": "str",
                    "compression_ratio": "float"
                },
                "use_cases": ["Text summarization", "Content condensation", "Key point extraction"],
                "prompt_template": """
Create summarization training data from this content.

CONTENT:
{content}

SUMMARY TYPE: {summary_type}

REQUIREMENTS:
- Generate {num_items} text-summary pairs
- Summaries should capture key information
- Include different summary lengths and styles
- Ensure summaries are accurate and coherent
- Calculate compression ratios

FORMAT: Return as JSON array:
[
  {{
    "source_text": "Original text segment",
    "summary": "Concise, accurate summary",
    "summary_type": "extractive/abstractive/bullet_points",
    "compression_ratio": 0.3,
    "key_points": ["point1", "point2"]
  }}
]
"""
            }
        }
    
    def _load_data_types(self) -> Dict[str, List[str]]:
        """Load data type configurations"""
        return {
            "classification_tasks": [
                "Sentiment Analysis", "Topic Classification", "Intent Detection",
                "Content Moderation", "Spam Detection", "Language Detection",
                "Emotion Recognition", "Urgency Classification"
            ],
            "instruction_types": [
                "Summarization", "Explanation", "Analysis", "Translation",
                "Question Answering", "Creative Writing", "Code Generation",
                "Data Extraction", "Comparison", "Evaluation"
            ],
            "summary_types": [
                "Extractive", "Abstractive", "Bullet Points", "Key Insights",
                "Executive Summary", "Technical Summary", "Lay Summary"
            ],
            "difficulty_levels": [
                "Beginner", "Intermediate", "Advanced", "Expert"
            ],
            "quality_metrics": [
                "Accuracy", "Relevance", "Completeness", "Clarity", "Consistency"
            ]
        }
    
    def get_available_formats(self) -> Dict[str, Dict[str, Any]]:
        """Get all available training data formats"""
        return self.format_templates
    
    def get_data_types(self) -> Dict[str, List[str]]:
        """Get all data type configurations"""
        return self.data_types
    
    def generate_training_data(self, chunks: List[Dict[str, Any]],
                             format_type: str,
                             config: Dict[str, Any] = None) -> List[TrainingDataItem]:
        """
        Generate training data from content chunks
        
        Args:
            chunks: List of content chunks
            format_type: Type of training data to generate
            config: Configuration options for generation
            
        Returns:
            List of TrainingDataItem objects
        """
        if config is None:
            config = {
                "num_items_per_chunk": 3,
                "quality_threshold": 0.7,
                "include_metadata": True
            }
        
        if format_type not in self.format_templates:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        results = []
        
        for chunk in chunks:
            try:
                chunk_results = self._generate_chunk_training_data(
                    chunk, format_type, config
                )
                results.extend(chunk_results)
                
                # Add small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error generating training data for chunk {chunk.get('id', 'unknown')}: {e}")
                # Create fallback data
                fallback_data = self._create_fallback_training_data(chunk, format_type, config)
                results.extend(fallback_data)
        
        return results
    
    def _generate_chunk_training_data(self, chunk: Dict[str, Any],
                                    format_type: str,
                                    config: Dict[str, Any]) -> List[TrainingDataItem]:
        """Generate training data for a single chunk"""
        
        template = self.format_templates[format_type]
        prompt = self._create_training_data_prompt(chunk, template, config)
        
        try:
            # Import OpenAI here to handle missing dependency gracefully
            import openai
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at creating high-quality training data for AI models. Always return valid JSON arrays as specified."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            
            # Parse response
            content = response.choices[0].message.content
            training_items = self._parse_training_data_response(content, chunk['id'], format_type)
            
            return training_items
            
        except ImportError:
            logger.warning("OpenAI not available, using fallback")
            return self._create_fallback_training_data(chunk, format_type, config)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._create_fallback_training_data(chunk, format_type, config)
    
    def _create_training_data_prompt(self, chunk: Dict[str, Any],
                                   template: Dict[str, Any],
                                   config: Dict[str, Any]) -> str:
        """Create prompt for training data generation"""
        
        content = chunk['text']
        num_items = config.get('num_items_per_chunk', 3)
        
        # Get template-specific configurations
        prompt_template = template['prompt_template']
        
        # Format prompt based on template type
        if template['name'] == "Text Classification Data":
            classification_task = config.get('classification_task', 'Topic Classification')
            labels = config.get('labels', ['positive', 'negative', 'neutral'])
            prompt = prompt_template.format(
                content=content,
                num_items=num_items,
                classification_task=classification_task,
                labels=', '.join(labels)
            )
        elif template['name'] == "Few-Shot Learning Examples":
            task_description = config.get('task_description', 'Extract key insights from text')
            prompt = prompt_template.format(
                content=content,
                num_items=num_items,
                task_description=task_description
            )
        elif template['name'] == "Summarization Training Data":
            summary_type = config.get('summary_type', 'Abstractive')
            prompt = prompt_template.format(
                content=content,
                num_items=num_items,
                summary_type=summary_type
            )
        else:
            # Default formatting
            prompt = prompt_template.format(
                content=content,
                num_items=num_items
            )
        
        return prompt
    
    def _parse_training_data_response(self, content: str, chunk_id: str, format_type: str) -> List[TrainingDataItem]:
        """Parse GPT response into TrainingDataItem objects"""
        
        training_items = []
        
        try:
            # Try to parse as JSON
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start != -1 and json_end != -1:
                json_content = content[json_start:json_end]
                parsed_data = json.loads(json_content)
                
                for i, item in enumerate(parsed_data):
                    # Create training data item based on format type
                    if format_type == "qa_pairs":
                        input_text = item.get('question', '')
                        output_text = item.get('answer', '')
                    elif format_type == "classification":
                        input_text = item.get('text', '')
                        output_text = item.get('label', '')
                    elif format_type == "instruction_following":
                        input_text = item.get('instruction', '')
                        output_text = item.get('output', '')
                    elif format_type == "dialogue_data":
                        input_text = item.get('user_message', '')
                        output_text = item.get('assistant_response', '')
                    elif format_type == "summarization_data":
                        input_text = item.get('source_text', '')
                        output_text = item.get('summary', '')
                    else:
                        input_text = str(item.get('input', ''))
                        output_text = str(item.get('output', ''))
                    
                    training_item = TrainingDataItem(
                        id=f"{chunk_id}_{format_type}_{i}",
                        format_type=format_type,
                        input_text=input_text,
                        output_text=output_text,
                        metadata=item,
                        quality_score=0.9
                    )
                    training_items.append(training_item)
            
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from GPT response for chunk {chunk_id}")
            # Create fallback items
            training_items = self._create_simple_fallback_items(content, chunk_id, format_type)
        
        return training_items
    
    def _create_fallback_training_data(self, chunk: Dict[str, Any],
                                     format_type: str,
                                     config: Dict[str, Any]) -> List[TrainingDataItem]:
        """Create fallback training data when API fails"""
        
        text = chunk['text']
        chunk_id = chunk['id']
        
        fallback_items = []
        
        if format_type == "qa_pairs":
            # Simple Q&A generation
            questions = [
                f"What is the main topic of this text?",
                f"What are the key points mentioned?",
                f"How would you summarize this content?"
            ]
            
            for i, question in enumerate(questions):
                item = TrainingDataItem(
                    id=f"{chunk_id}_qa_fallback_{i}",
                    format_type="qa_pairs",
                    input_text=question,
                    output_text=f"Based on the content: {text[:200]}...",
                    metadata={"question": question, "context": text},
                    quality_score=0.6
                )
                fallback_items.append(item)
        
        elif format_type == "classification":
            # Simple classification data
            item = TrainingDataItem(
                id=f"{chunk_id}_classification_fallback",
                format_type="classification",
                input_text=text[:500],
                output_text="informational",
                metadata={"text": text[:500], "label": "informational"},
                quality_score=0.6
            )
            fallback_items.append(item)
        
        else:
            # Generic fallback
            item = TrainingDataItem(
                id=f"{chunk_id}_{format_type}_fallback",
                format_type=format_type,
                input_text=text[:300],
                output_text=f"Processed content from {format_type}",
                metadata={"original_text": text},
                quality_score=0.5
            )
            fallback_items.append(item)
        
        return fallback_items
    
    def _create_simple_fallback_items(self, content: str, chunk_id: str, format_type: str) -> List[TrainingDataItem]:
        """Create simple fallback items from unparseable content"""
        
        # Try to extract any useful information from the content
        lines = content.split('\n')
        items = []
        
        for i, line in enumerate(lines[:3]):  # Max 3 items
            if line.strip():
                item = TrainingDataItem(
                    id=f"{chunk_id}_{format_type}_simple_{i}",
                    format_type=format_type,
                    input_text=line.strip()[:200],
                    output_text="Generated from content",
                    metadata={"source_line": line.strip()},
                    quality_score=0.5
                )
                items.append(item)
        
        return items
    
    def export_training_data(self, training_items: List[TrainingDataItem],
                           export_format: str = "json") -> str:
        """
        Export training data in specified format
        
        Args:
            training_items: List of training data items
            export_format: Format for export (json, jsonl, csv)
            
        Returns:
            Formatted string ready for export
        """
        
        if export_format.lower() == "json":
            return json.dumps([item.to_dict() for item in training_items], indent=2)
        
        elif export_format.lower() == "jsonl":
            return '\n'.join([item.to_jsonl() for item in training_items])
        
        elif export_format.lower() == "csv":
            # Create CSV format
            import csv
            import io
            
            output = io.StringIO()
            if training_items:
                fieldnames = training_items[0].to_dict().keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for item in training_items:
                    writer.writerow(item.to_dict())
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
    
    def validate_training_data(self, training_items: List[TrainingDataItem]) -> Dict[str, Any]:
        """Validate quality of generated training data"""
        
        if not training_items:
            return {"valid": False, "errors": ["No training data generated"]}
        
        validation_results = {
            "valid": True,
            "total_items": len(training_items),
            "average_quality": sum(item.quality_score for item in training_items) / len(training_items),
            "format_distribution": {},
            "quality_distribution": {"high": 0, "medium": 0, "low": 0},
            "errors": []
        }
        
        # Analyze format distribution
        for item in training_items:
            format_type = item.format_type
            validation_results["format_distribution"][format_type] = validation_results["format_distribution"].get(format_type, 0) + 1
            
            # Analyze quality distribution
            if item.quality_score >= 0.8:
                validation_results["quality_distribution"]["high"] += 1
            elif item.quality_score >= 0.6:
                validation_results["quality_distribution"]["medium"] += 1
            else:
                validation_results["quality_distribution"]["low"] += 1
            
            # Check for basic validation errors
            if not item.input_text.strip():
                validation_results["errors"].append(f"Empty input text in item {item.id}")
            if not item.output_text.strip():
                validation_results["errors"].append(f"Empty output text in item {item.id}")
        
        if validation_results["errors"]:
            validation_results["valid"] = False
        
        return validation_results

