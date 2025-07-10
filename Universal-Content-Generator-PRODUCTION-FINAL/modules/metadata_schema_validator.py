"""
Metadata Schema Validator Module
===============================

Provides comprehensive schema validation for the Enhanced Universal AI Training Data Creator.
Uses Pydantic models to ensure data integrity and downstream compatibility with fine-tuning pipelines.

Features:
- Pydantic schema models for all export formats
- Validation of training data structure
- Metadata consistency checking
- Schema version management
- Export format compliance
"""

from pydantic import BaseModel, Field, validator, ValidationError
from typing import Dict, Any, List, Optional, Union, Literal, Tuple
from datetime import datetime
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Enumeration of supported content types"""
    QA = "qa"
    DIALOGUE = "dialogue"
    MONOLOGUE = "monologue"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class SpiritualTone(str, Enum):
    """Enumeration of supported spiritual tones"""
    ZEN_BUDDHISM = "zen_buddhism"
    ADVAITA_VEDANTA = "advaita_vedanta"
    CHRISTIAN_MYSTICISM = "christian_mysticism"
    SUFI_MYSTICISM = "sufi_mysticism"
    MINDFULNESS_MEDITATION = "mindfulness_meditation"
    UNIVERSAL_WISDOM = "universal_wisdom"
    CUSTOM = "custom"


class ExportFormat(str, Enum):
    """Enumeration of supported export formats"""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    XLSX = "xlsx"
    TXT = "txt"
    ZIP = "zip"


class QualityMetrics(BaseModel):
    """Schema for quality assessment metrics"""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score (0-1)")
    semantic_similarity: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity to original")
    hallucination_score: float = Field(..., ge=0.0, le=1.0, description="Hallucination detection score")
    readability_score: float = Field(..., ge=0.0, le=1.0, description="Content readability score")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Content coherence score")
    tone_consistency: Optional[float] = Field(None, ge=0.0, le=1.0, description="Tone consistency score")
    length_ratio: float = Field(..., gt=0.0, description="Enhanced to original length ratio")
    
    @validator('length_ratio')
    def validate_length_ratio(cls, v):
        if v > 10.0:  # Sanity check - enhanced shouldn't be >10x original
            raise ValueError("Length ratio too high - possible processing error")
        return v


class ProcessingMetadata(BaseModel):
    """Schema for processing metadata"""
    extraction_method: str = Field(..., description="Method used for content extraction")
    enhancement_model: str = Field(..., description="AI model used for enhancement")
    processing_time: float = Field(..., ge=0.0, description="Total processing time in seconds")
    api_cost: Optional[float] = Field(None, ge=0.0, description="API cost in USD")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    schema_version: str = Field(default="1.0", description="Schema version")


class TrainingDataItem(BaseModel):
    """Schema for individual training data items"""
    input: str = Field(..., min_length=1, description="Original input text")
    output: str = Field(..., min_length=1, description="Enhanced output text")
    content_type: ContentType = Field(..., description="Detected content type")
    spiritual_tone: SpiritualTone = Field(..., description="Applied spiritual tone")
    quality_metrics: QualityMetrics = Field(..., description="Quality assessment metrics")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    manual_review: Optional[Dict[str, Any]] = Field(None, description="Manual review data")
    custom_metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata fields")
    
    @validator('input', 'output')
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError("Text content cannot be empty or whitespace only")
        if len(v) > 100000:  # 100k character limit
            raise ValueError("Text content exceeds maximum length")
        return v.strip()
    
    @validator('output')
    def validate_enhancement_quality(cls, v, values):
        if 'input' in values:
            input_text = values['input']
            # Basic sanity checks
            if len(v) < len(input_text) * 0.1:  # Enhanced shouldn't be <10% of original
                raise ValueError("Enhanced content too short compared to original")
            if len(v) > len(input_text) * 20:  # Enhanced shouldn't be >20x original
                raise ValueError("Enhanced content too long compared to original")
        return v


class DatasetMetadata(BaseModel):
    """Schema for overall dataset metadata"""
    dataset_name: str = Field(..., description="Name of the dataset")
    description: Optional[str] = Field(None, description="Dataset description")
    total_items: int = Field(..., ge=0, description="Total number of items")
    content_type_distribution: Dict[ContentType, int] = Field(..., description="Distribution of content types")
    spiritual_tone_distribution: Dict[SpiritualTone, int] = Field(..., description="Distribution of spiritual tones")
    quality_summary: Dict[str, float] = Field(..., description="Summary of quality metrics")
    processing_summary: ProcessingMetadata = Field(..., description="Overall processing metadata")
    export_format: ExportFormat = Field(..., description="Export format used")
    created_by: str = Field(default="Enhanced Universal AI Training Data Creator", description="Creator application")
    
    @validator('total_items')
    def validate_item_count(cls, v, values):
        if 'content_type_distribution' in values:
            distribution_total = sum(values['content_type_distribution'].values())
            if v != distribution_total:
                raise ValueError("Total items doesn't match content type distribution sum")
        return v


class ExportPackage(BaseModel):
    """Schema for complete export package"""
    metadata: DatasetMetadata = Field(..., description="Dataset metadata")
    training_data: List[TrainingDataItem] = Field(..., description="Training data items")
    validation_report: Optional[Dict[str, Any]] = Field(None, description="Validation report")
    
    @validator('training_data')
    def validate_training_data_consistency(cls, v, values):
        if not v:
            raise ValueError("Training data cannot be empty")
        
        if 'metadata' in values:
            metadata = values['metadata']
            if len(v) != metadata.total_items:
                raise ValueError("Training data count doesn't match metadata")
        
        return v


class MetadataSchemaValidator:
    """Metadata schema validator with comprehensive validation capabilities"""
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
        
    def validate_training_item(self, item_data: Dict[str, Any]) -> Tuple[bool, TrainingDataItem, List[str]]:
        """Validate individual training data item"""
        try:
            validated_item = TrainingDataItem(**item_data)
            return True, validated_item, []
        except ValidationError as e:
            error_messages = [f"{error['loc']}: {error['msg']}" for error in e.errors()]
            return False, None, error_messages
    
    def validate_dataset_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, DatasetMetadata, List[str]]:
        """Validate dataset metadata"""
        try:
            validated_metadata = DatasetMetadata(**metadata)
            return True, validated_metadata, []
        except ValidationError as e:
            error_messages = [f"{error['loc']}: {error['msg']}" for error in e.errors()]
            return False, None, error_messages
    
    def validate_export_package(self, package_data: Dict[str, Any]) -> Tuple[bool, ExportPackage, List[str]]:
        """Validate complete export package"""
        try:
            validated_package = ExportPackage(**package_data)
            return True, validated_package, []
        except ValidationError as e:
            error_messages = [f"{error['loc']}: {error['msg']}" for error in e.errors()]
            return False, None, error_messages
    
    def validate_batch_training_data(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate batch of training data items"""
        validation_results = {
            'total_items': len(training_data),
            'valid_items': 0,
            'invalid_items': 0,
            'validation_errors': [],
            'validated_data': []
        }
        
        for i, item_data in enumerate(training_data):
            is_valid, validated_item, errors = self.validate_training_item(item_data)
            
            if is_valid:
                validation_results['valid_items'] += 1
                validation_results['validated_data'].append(validated_item.dict())
            else:
                validation_results['invalid_items'] += 1
                validation_results['validation_errors'].append({
                    'item_index': i,
                    'errors': errors
                })
        
        validation_results['validation_rate'] = (
            validation_results['valid_items'] / validation_results['total_items'] 
            if validation_results['total_items'] > 0 else 0
        )
        
        return validation_results
    
    def generate_schema_documentation(self) -> Dict[str, Any]:
        """Generate comprehensive schema documentation"""
        return {
            'schema_version': '1.0',
            'models': {
                'TrainingDataItem': {
                    'description': 'Individual training data item with input/output pair',
                    'required_fields': list(TrainingDataItem.__fields__.keys()),
                    'schema': TrainingDataItem.schema()
                },
                'DatasetMetadata': {
                    'description': 'Metadata for the complete dataset',
                    'required_fields': list(DatasetMetadata.__fields__.keys()),
                    'schema': DatasetMetadata.schema()
                },
                'ExportPackage': {
                    'description': 'Complete export package with data and metadata',
                    'required_fields': list(ExportPackage.__fields__.keys()),
                    'schema': ExportPackage.schema()
                }
            },
            'enums': {
                'ContentType': [e.value for e in ContentType],
                'SpiritualTone': [e.value for e in SpiritualTone],
                'ExportFormat': [e.value for e in ExportFormat]
            },
            'validation_rules': {
                'text_length': 'Text fields must be 1-100,000 characters',
                'quality_scores': 'All quality scores must be between 0.0 and 1.0',
                'length_ratio': 'Enhanced text should be 0.1x to 20x original length',
                'consistency': 'Metadata totals must match actual data counts'
            }
        }
    
    def create_sample_data(self) -> Dict[str, Any]:
        """Create sample data conforming to schema"""
        sample_item = TrainingDataItem(
            input="What is the nature of consciousness?",
            output="Consciousness is the fundamental awareness that underlies all experience, the pure knowing that remains constant through all changing phenomena.",
            content_type=ContentType.QA,
            spiritual_tone=SpiritualTone.ADVAITA_VEDANTA,
            quality_metrics=QualityMetrics(
                overall_score=0.85,
                semantic_similarity=0.92,
                hallucination_score=0.05,
                readability_score=0.78,
                coherence_score=0.88,
                tone_consistency=0.91,
                length_ratio=1.45
            ),
            processing_metadata=ProcessingMetadata(
                extraction_method="text_extraction",
                enhancement_model="gpt-4",
                processing_time=2.34,
                api_cost=0.0023,
                token_usage={"input": 12, "output": 28},
                created_at=datetime.now(),
                schema_version="1.0"
            )
        )
        
        sample_metadata = DatasetMetadata(
            dataset_name="Sample Consciousness Dataset",
            description="Sample dataset for consciousness-related Q&A",
            total_items=1,
            content_type_distribution={ContentType.QA: 1},
            spiritual_tone_distribution={SpiritualTone.ADVAITA_VEDANTA: 1},
            quality_summary={
                "avg_overall_score": 0.85,
                "avg_semantic_similarity": 0.92,
                "avg_hallucination_score": 0.05
            },
            processing_summary=ProcessingMetadata(
                extraction_method="batch_processing",
                enhancement_model="gpt-4",
                processing_time=2.34,
                api_cost=0.0023,
                created_at=datetime.now()
            ),
            export_format=ExportFormat.JSON
        )
        
        sample_package = ExportPackage(
            metadata=sample_metadata,
            training_data=[sample_item]
        )
        
        return sample_package.dict()
    
    def validate_for_fine_tuning(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate data specifically for fine-tuning compatibility"""
        fine_tuning_report = {
            'compatible': True,
            'total_items': len(training_data),
            'issues': [],
            'recommendations': [],
            'statistics': {}
        }
        
        if len(training_data) < 10:
            fine_tuning_report['issues'].append("Dataset too small for effective fine-tuning (minimum 10 items recommended)")
            fine_tuning_report['compatible'] = False
        
        # Check input/output length distribution
        input_lengths = [len(item.get('input', '')) for item in training_data]
        output_lengths = [len(item.get('output', '')) for item in training_data]
        
        avg_input_length = sum(input_lengths) / len(input_lengths) if input_lengths else 0
        avg_output_length = sum(output_lengths) / len(output_lengths) if output_lengths else 0
        
        fine_tuning_report['statistics'] = {
            'avg_input_length': avg_input_length,
            'avg_output_length': avg_output_length,
            'length_ratio': avg_output_length / avg_input_length if avg_input_length > 0 else 0
        }
        
        # Check for consistency
        content_types = set(item.get('content_type') for item in training_data)
        spiritual_tones = set(item.get('spiritual_tone') for item in training_data)
        
        if len(content_types) > 3:
            fine_tuning_report['recommendations'].append("Consider grouping by content type for more focused fine-tuning")
        
        if len(spiritual_tones) > 2:
            fine_tuning_report['recommendations'].append("Consider training separate models for different spiritual tones")
        
        # Check quality thresholds
        quality_scores = [
            item.get('quality_metrics', {}).get('overall_score', 0) 
            for item in training_data
        ]
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        low_quality_count = sum(1 for score in quality_scores if score < 0.7)
        
        if avg_quality < 0.75:
            fine_tuning_report['issues'].append(f"Average quality score ({avg_quality:.2f}) below recommended threshold (0.75)")
            fine_tuning_report['compatible'] = False
        
        if low_quality_count > len(training_data) * 0.1:  # >10% low quality
            fine_tuning_report['recommendations'].append(f"Consider removing {low_quality_count} low-quality items")
        
        return fine_tuning_report
    
    def export_schema_files(self, output_dir: str = "."):
        """Export schema files for external validation"""
        import os
        
        # Create schema directory
        schema_dir = os.path.join(output_dir, "schemas")
        os.makedirs(schema_dir, exist_ok=True)
        
        # Export individual schemas
        schemas = {
            'training_data_item.json': TrainingDataItem.schema(),
            'dataset_metadata.json': DatasetMetadata.schema(),
            'export_package.json': ExportPackage.schema()
        }
        
        for filename, schema in schemas.items():
            filepath = os.path.join(schema_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(schema, f, indent=2)
        
        # Export documentation
        doc_filepath = os.path.join(schema_dir, 'schema_documentation.json')
        with open(doc_filepath, 'w') as f:
            json.dump(self.generate_schema_documentation(), f, indent=2)
        
        # Export sample data
        sample_filepath = os.path.join(schema_dir, 'sample_data.json')
        with open(sample_filepath, 'w') as f:
            json.dump(self.create_sample_data(), f, indent=2, default=str)
        
        logger.info(f"Schema files exported to {schema_dir}")
        
        return schema_dir


# Global metadata schema validator instance
metadata_validator = MetadataSchemaValidator()

