#!/usr/bin/env python3
"""
Validation Engine for Universal Text-to-Dialogue AI
=================================================

Comprehensive validation system for training data format validation,
content integrity checks, and export preview validation.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

@dataclass
class ValidationResult:
    """Result of a validation check"""
    is_valid: bool
    score: float
    issues: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]

class TrainingDataValidator:
    """Validates training data format and quality"""
    
    def __init__(self):
        self.required_fields = ["chunk_id", "original_content", "gpt_output"]
        self.optional_fields = ["word_count", "quality_score", "processing_type"]
        
    def validate_jsonl_format(self, data: List[Dict[str, Any]]) -> ValidationResult:
        """Validate JSONL format for training data"""
        issues = []
        warnings = []
        valid_entries = 0
        
        for i, entry in enumerate(data):
            # Check required fields
            missing_fields = [field for field in self.required_fields if field not in entry]
            if missing_fields:
                issues.append(f"Entry {i}: Missing required fields: {missing_fields}")
                continue
            
            # Check field types and content
            if not isinstance(entry.get("chunk_id"), str) or not entry["chunk_id"].strip():
                issues.append(f"Entry {i}: Invalid chunk_id")
                continue
                
            if not isinstance(entry.get("original_content"), str) or len(entry["original_content"].strip()) < 10:
                issues.append(f"Entry {i}: Invalid or too short original_content")
                continue
                
            if not isinstance(entry.get("gpt_output"), str) or len(entry["gpt_output"].strip()) < 10:
                issues.append(f"Entry {i}: Invalid or too short gpt_output")
                continue
            
            # Check optional fields
            if "word_count" in entry and not isinstance(entry["word_count"], (int, float)):
                warnings.append(f"Entry {i}: Invalid word_count type")
                
            if "quality_score" in entry:
                score = entry["quality_score"]
                if not isinstance(score, (int, float)) or not (0 <= score <= 1):
                    warnings.append(f"Entry {i}: Invalid quality_score (should be 0-1)")
            
            valid_entries += 1
        
        # Calculate validation score
        total_entries = len(data)
        validation_score = valid_entries / total_entries if total_entries > 0 else 0
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=validation_score,
            issues=issues,
            warnings=warnings,
            metadata={
                "total_entries": total_entries,
                "valid_entries": valid_entries,
                "invalid_entries": total_entries - valid_entries,
                "validation_timestamp": datetime.now().isoformat()
            }
        )
    
    def validate_dialogue_format(self, gpt_output: str) -> ValidationResult:
        """Validate dialogue format in GPT output"""
        issues = []
        warnings = []
        
        # Check for dialogue markers
        dialogue_patterns = [
            r'\*\*Teacher:\*\*',
            r'\*\*Student:\*\*',
            r'Teacher:',
            r'Student:',
            r'Q:',
            r'A:'
        ]
        
        has_dialogue_markers = any(re.search(pattern, gpt_output, re.IGNORECASE) for pattern in dialogue_patterns)
        
        if not has_dialogue_markers:
            warnings.append("No clear dialogue markers found (Teacher/Student, Q/A)")
        
        # Check for balanced dialogue
        teacher_count = len(re.findall(r'\*\*Teacher:\*\*|Teacher:', gpt_output, re.IGNORECASE))
        student_count = len(re.findall(r'\*\*Student:\*\*|Student:', gpt_output, re.IGNORECASE))
        
        if teacher_count > 0 and student_count > 0:
            if abs(teacher_count - student_count) > 2:
                warnings.append(f"Unbalanced dialogue: {teacher_count} teacher vs {student_count} student entries")
        
        # Check minimum length
        if len(gpt_output.strip()) < 100:
            issues.append("GPT output too short for meaningful dialogue")
        
        # Check for proper formatting
        if not re.search(r'[.!?]', gpt_output):
            issues.append("No proper sentence endings found")
        
        # Calculate score based on dialogue quality
        score = 1.0
        if issues:
            score -= 0.5 * len(issues)
        if warnings:
            score -= 0.1 * len(warnings)
        score = max(0, min(1, score))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            metadata={
                "teacher_count": teacher_count,
                "student_count": student_count,
                "has_dialogue_markers": has_dialogue_markers,
                "content_length": len(gpt_output)
            }
        )

class ContentIntegrityChecker:
    """Checks content integrity and completeness"""
    
    def validate_chunk_completeness(self, chunk_data: Dict[str, Any]) -> ValidationResult:
        """Validate that chunk content is complete and coherent"""
        issues = []
        warnings = []
        
        original_content = chunk_data.get("original_content", "")
        gpt_output = chunk_data.get("gpt_output", "")
        
        # Handle malformed data gracefully
        if not isinstance(original_content, str):
            issues.append(f"Original content must be string, got {type(original_content).__name__}")
            original_content = str(original_content) if original_content is not None else ""
        
        if not isinstance(gpt_output, str):
            issues.append(f"GPT output must be string, got {type(gpt_output).__name__}")
            gpt_output = str(gpt_output) if gpt_output is not None else ""
        
        # Check content lengths
        original_words = len(original_content.split())
        output_words = len(gpt_output.split())
        
        if original_words < 50:
            warnings.append("Original content is very short (< 50 words)")
        elif original_words > 1000:
            warnings.append("Original content is very long (> 1000 words)")
        
        if output_words < 50:
            issues.append("GPT output is too short (< 50 words)")
        elif output_words > 2000:
            warnings.append("GPT output is very long (> 2000 words)")
        
        # Check for content truncation
        if original_content.endswith("...") or gpt_output.endswith("..."):
            warnings.append("Content appears to be truncated")
        
        # Check for incomplete sentences
        if original_content.strip() and not original_content.strip().endswith(('.', '!', '?', '"', "'")):
            warnings.append("Original content doesn't end with proper punctuation")
        
        if gpt_output.strip() and not gpt_output.strip().endswith(('.', '!', '?', '"', "'")):
            warnings.append("GPT output doesn't end with proper punctuation")
        
        # Check for coherence (basic)
        original_sentences = len([s for s in original_content.split('.') if s.strip()])
        output_sentences = len([s for s in gpt_output.split('.') if s.strip()])
        
        if original_sentences > 0 and output_sentences > 0:
            sentence_ratio = output_sentences / original_sentences
            if sentence_ratio > 5:
                warnings.append("GPT output significantly longer than original (possible repetition)")
            elif sentence_ratio < 0.5:
                warnings.append("GPT output much shorter than original (possible information loss)")
        
        # Calculate integrity score
        score = 1.0
        if issues:
            score -= 0.3 * len(issues)
        if warnings:
            score -= 0.1 * len(warnings)
        score = max(0, min(1, score))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            metadata={
                "original_words": original_words,
                "output_words": output_words,
                "original_sentences": original_sentences,
                "output_sentences": output_sentences,
                "word_ratio": output_words / max(original_words, 1),
                "sentence_ratio": output_sentences / max(original_sentences, 1)
            }
        )
    
    def validate_content_quality(self, content: str) -> ValidationResult:
        """Validate content quality metrics"""
        issues = []
        warnings = []
        
        # Check for basic quality indicators
        words = content.split()
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        
        # Check vocabulary diversity
        unique_words = len(set(word.lower().strip('.,!?";') for word in words))
        vocabulary_ratio = unique_words / max(len(words), 1)
        
        if vocabulary_ratio < 0.3:
            warnings.append("Low vocabulary diversity (possible repetition)")
        
        # Check sentence length variation
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            
            if avg_length < 5:
                warnings.append("Very short average sentence length")
            elif avg_length > 30:
                warnings.append("Very long average sentence length")
            
            # Check for sentence length variation
            if len(set(sentence_lengths)) < len(sentence_lengths) * 0.3:
                warnings.append("Low sentence length variation")
        
        # Check for common issues
        if content.count('...') > 3:
            warnings.append("Excessive use of ellipsis")
        
        if len(re.findall(r'\b(\w+)\s+\1\b', content, re.IGNORECASE)) > 2:
            warnings.append("Repeated words detected")
        
        # Calculate quality score
        score = 1.0
        if vocabulary_ratio < 0.3:
            score -= 0.2
        if warnings:
            score -= 0.1 * len(warnings)
        if issues:
            score -= 0.3 * len(issues)
        score = max(0, min(1, score))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            metadata={
                "word_count": len(words),
                "unique_words": unique_words,
                "vocabulary_ratio": vocabulary_ratio,
                "sentence_count": len(sentences),
                "avg_sentence_length": avg_length if sentences else 0
            }
        )

class ExportPreviewValidator:
    """Validates export data before download"""
    
    def validate_export_data(self, export_data: List[Dict[str, Any]], export_format: str) -> ValidationResult:
        """Validate export data before generating download"""
        issues = []
        warnings = []
        
        if not export_data:
            issues.append("No data to export")
            return ValidationResult(
                is_valid=False,
                score=0.0,
                issues=issues,
                warnings=warnings,
                metadata={"total_entries": 0}
            )
        
        # Validate based on export format
        if export_format.lower() == "jsonl":
            # JSONL specific validation
            for i, entry in enumerate(export_data):
                try:
                    json.dumps(entry)
                except (TypeError, ValueError) as e:
                    issues.append(f"Entry {i}: Not JSON serializable - {str(e)}")
        
        elif export_format.lower() == "csv":
            # CSV specific validation
            try:
                df = pd.DataFrame(export_data)
                # Check for problematic characters
                for col in df.columns:
                    if df[col].dtype == 'object':
                        problematic_entries = df[df[col].astype(str).str.contains(r'["\n\r]', na=False)]
                        if not problematic_entries.empty:
                            warnings.append(f"Column '{col}' contains quotes or newlines that may affect CSV format")
            except Exception as e:
                issues.append(f"Cannot convert to CSV format: {str(e)}")
        
        # General validation
        total_size = 0
        for entry in export_data:
            entry_size = len(str(entry))
            total_size += entry_size
            
            if entry_size > 100000:  # 100KB per entry
                warnings.append(f"Large entry detected: {entry.get('chunk_id', 'unknown')} ({entry_size} bytes)")
        
        # Check total size
        if total_size > 50 * 1024 * 1024:  # 50MB
            warnings.append(f"Large export size: {total_size / (1024*1024):.1f}MB")
        
        # Calculate validation score
        score = 1.0
        if issues:
            score -= 0.5 * len(issues)
        if warnings:
            score -= 0.1 * len(warnings)
        score = max(0, min(1, score))
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=score,
            issues=issues,
            warnings=warnings,
            metadata={
                "total_entries": len(export_data),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024*1024),
                "export_format": export_format,
                "validation_timestamp": datetime.now().isoformat()
            }
        )
    
    def generate_export_preview(self, export_data: List[Dict[str, Any]], max_entries: int = 3) -> Dict[str, Any]:
        """Generate a preview of export data"""
        preview_data = export_data[:max_entries]
        
        # Calculate statistics
        total_entries = len(export_data)
        total_words = sum(
            len(str(entry.get("original_content", "")).split()) + 
            len(str(entry.get("gpt_output", "")).split())
            for entry in export_data
        )
        
        avg_quality = 0
        quality_scores = [entry.get("quality_score") for entry in export_data if entry.get("quality_score") is not None]
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
        
        return {
            "preview_entries": preview_data,
            "statistics": {
                "total_entries": total_entries,
                "total_words": total_words,
                "average_quality_score": avg_quality,
                "entries_with_quality_scores": len(quality_scores),
                "preview_count": len(preview_data)
            },
            "sample_fields": list(preview_data[0].keys()) if preview_data else [],
            "estimated_file_size_mb": sum(len(str(entry)) for entry in export_data) / (1024*1024)
        }

class ComprehensiveValidator:
    """Main validator that combines all validation types"""
    
    def __init__(self):
        self.training_validator = TrainingDataValidator()
        self.integrity_checker = ContentIntegrityChecker()
        self.export_validator = ExportPreviewValidator()
    
    def validate_complete_dataset(self, export_data: List[Dict[str, Any]]) -> Dict[str, ValidationResult]:
        """Run comprehensive validation on complete dataset"""
        results = {}
        
        # 1. Training data format validation
        results["training_format"] = self.training_validator.validate_jsonl_format(export_data)
        
        # 2. Content integrity validation
        integrity_results = []
        for entry in export_data:
            result = self.integrity_checker.validate_chunk_completeness(entry)
            integrity_results.append(result)
        
        # Aggregate integrity results
        avg_integrity_score = sum(r.score for r in integrity_results) / len(integrity_results) if integrity_results else 0
        all_integrity_issues = []
        all_integrity_warnings = []
        
        for i, result in enumerate(integrity_results):
            all_integrity_issues.extend([f"Entry {i}: {issue}" for issue in result.issues])
            all_integrity_warnings.extend([f"Entry {i}: {warning}" for warning in result.warnings])
        
        results["content_integrity"] = ValidationResult(
            is_valid=all(r.is_valid for r in integrity_results),
            score=avg_integrity_score,
            issues=all_integrity_issues,
            warnings=all_integrity_warnings,
            metadata={"entries_validated": len(integrity_results)}
        )
        
        # 3. Dialogue format validation
        dialogue_results = []
        for entry in export_data:
            if entry.get("gpt_output"):
                result = self.training_validator.validate_dialogue_format(entry["gpt_output"])
                dialogue_results.append(result)
        
        if dialogue_results:
            avg_dialogue_score = sum(r.score for r in dialogue_results) / len(dialogue_results)
            all_dialogue_issues = []
            all_dialogue_warnings = []
            
            for i, result in enumerate(dialogue_results):
                all_dialogue_issues.extend([f"Entry {i}: {issue}" for issue in result.issues])
                all_dialogue_warnings.extend([f"Entry {i}: {warning}" for warning in result.warnings])
            
            results["dialogue_format"] = ValidationResult(
                is_valid=all(r.is_valid for r in dialogue_results),
                score=avg_dialogue_score,
                issues=all_dialogue_issues,
                warnings=all_dialogue_warnings,
                metadata={"entries_validated": len(dialogue_results)}
            )
        
        return results
    
    def get_validation_summary(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate a summary of all validation results"""
        total_issues = sum(len(result.issues) for result in validation_results.values())
        total_warnings = sum(len(result.warnings) for result in validation_results.values())
        avg_score = sum(result.score for result in validation_results.values()) / len(validation_results)
        
        overall_valid = all(result.is_valid for result in validation_results.values())
        
        return {
            "overall_valid": overall_valid,
            "overall_score": avg_score,
            "total_issues": total_issues,
            "total_warnings": total_warnings,
            "validation_categories": len(validation_results),
            "category_scores": {name: result.score for name, result in validation_results.items()},
            "recommendation": self._get_recommendation(avg_score, total_issues, total_warnings)
        }
    
    def _get_recommendation(self, score: float, issues: int, warnings: int) -> str:
        """Get recommendation based on validation results"""
        if score >= 0.9 and issues == 0:
            return "✅ Excellent quality - Ready for production use"
        elif score >= 0.8 and issues <= 2:
            return "🟢 Good quality - Minor improvements recommended"
        elif score >= 0.6 and issues <= 5:
            return "🟡 Acceptable quality - Review and fix issues before use"
        elif score >= 0.4:
            return "🟠 Poor quality - Significant improvements needed"
        else:
            return "🔴 Unacceptable quality - Major revision required"

