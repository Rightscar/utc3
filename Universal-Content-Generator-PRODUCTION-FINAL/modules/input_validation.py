"""
Input Validation Module
======================

Provides comprehensive input validation to prevent list vs string issues
and other common data type problems in the Universal Fine-Tune Data System.

Features:
- File upload validation
- Environment variable validation
- Data type conversion and validation
- Safe string/list handling
"""

import os
import logging
from typing import Any, List, Union, Optional, Dict
import streamlit as st

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation for the application"""
    
    @staticmethod
    def validate_uploaded_files(uploaded_files: Any) -> List[Any]:
        """
        Safely validate and normalize uploaded files from Streamlit file_uploader
        
        Args:
            uploaded_files: Output from st.file_uploader (can be None, single file, or list)
            
        Returns:
            List of valid uploaded files (empty list if none)
        """
        if uploaded_files is None:
            return []
        
        # Handle single file (when accept_multiple_files=False)
        if not isinstance(uploaded_files, list):
            return [uploaded_files] if uploaded_files is not None else []
        
        # Handle multiple files (when accept_multiple_files=True)
        return [f for f in uploaded_files if f is not None]
    
    @staticmethod
    def safe_env_var(var_name: str, default: str = "", expected_type: str = "string") -> Union[str, bool, int]:
        """
        Safely get environment variable with type validation
        
        Args:
            var_name: Environment variable name
            default: Default value if not found
            expected_type: Expected type ("string", "boolean", "integer")
            
        Returns:
            Validated environment variable value
        """
        try:
            value = os.getenv(var_name, default)
            
            if expected_type == "boolean":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "on")
                return bool(value)
            
            elif expected_type == "integer":
                return int(value) if value else 0
            
            else:  # string
                return str(value) if value is not None else default
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid environment variable {var_name}: {e}, using default: {default}")
            return default
    
    @staticmethod
    def safe_string_split(text: Any, delimiter: str = " ", max_splits: int = -1) -> List[str]:
        """
        Safely split text, handling various input types
        
        Args:
            text: Text to split (can be string, list, or other)
            delimiter: Split delimiter
            max_splits: Maximum number of splits
            
        Returns:
            List of string parts
        """
        try:
            if text is None:
                return []
            
            if isinstance(text, list):
                # If already a list, convert items to strings
                return [str(item) for item in text]
            
            if isinstance(text, str):
                if max_splits > 0:
                    return text.split(delimiter, max_splits)
                else:
                    return text.split(delimiter)
            
            # Convert other types to string first
            return str(text).split(delimiter)
            
        except Exception as e:
            logger.warning(f"Error splitting text: {e}")
            return [str(text)] if text is not None else []
    
    @staticmethod
    def safe_string_join(items: Any, delimiter: str = " ") -> str:
        """
        Safely join items into a string, handling various input types
        
        Args:
            items: Items to join (can be list, string, or other)
            delimiter: Join delimiter
            
        Returns:
            Joined string
        """
        try:
            if items is None:
                return ""
            
            if isinstance(items, str):
                return items
            
            if isinstance(items, list):
                return delimiter.join(str(item) for item in items)
            
            # Convert other types to string
            return str(items)
            
        except Exception as e:
            logger.warning(f"Error joining items: {e}")
            return str(items) if items is not None else ""
    
    @staticmethod
    def validate_selectbox_output(value: Any, options: List[str], default: str = None) -> str:
        """
        Validate selectbox output to ensure it's a string
        
        Args:
            value: Output from st.selectbox
            options: Valid options list
            default: Default value if validation fails
            
        Returns:
            Valid string option
        """
        try:
            if value is None:
                return default or (options[0] if options else "")
            
            if isinstance(value, list):
                # If somehow a list is returned, take the first item
                value = value[0] if value else default
            
            value_str = str(value)
            
            # Ensure the value is in the options
            if value_str in options:
                return value_str
            
            # If not in options, return default or first option
            return default or (options[0] if options else "")
            
        except Exception as e:
            logger.warning(f"Error validating selectbox output: {e}")
            return default or (options[0] if options else "")
    
    @staticmethod
    def validate_multiselect_output(value: Any, options: List[str]) -> List[str]:
        """
        Validate multiselect output to ensure it's a list of strings
        
        Args:
            value: Output from st.multiselect
            options: Valid options list
            
        Returns:
            List of valid string options
        """
        try:
            if value is None:
                return []
            
            if isinstance(value, str):
                # If somehow a string is returned, convert to list
                return [value] if value in options else []
            
            if isinstance(value, list):
                # Validate each item in the list
                return [str(item) for item in value if str(item) in options]
            
            # Convert other types
            value_str = str(value)
            return [value_str] if value_str in options else []
            
        except Exception as e:
            logger.warning(f"Error validating multiselect output: {e}")
            return []
    
    @staticmethod
    def safe_file_path(path: Any) -> str:
        """
        Safely convert path to string, handling various input types
        
        Args:
            path: File path (can be string, Path object, or other)
            
        Returns:
            Valid string path
        """
        try:
            if path is None:
                return ""
            
            if isinstance(path, list):
                # If somehow a list, take the first item
                path = path[0] if path else ""
            
            return str(path)
            
        except Exception as e:
            logger.warning(f"Error converting path: {e}")
            return ""
    
    @staticmethod
    def validate_content_data(content: Any) -> str:
        """
        Validate content data to ensure it's a string
        
        Args:
            content: Content data (can be string, list, dict, or other)
            
        Returns:
            Valid string content
        """
        try:
            if content is None:
                return ""
            
            if isinstance(content, str):
                return content
            
            if isinstance(content, list):
                # Join list items with newlines
                return "\n".join(str(item) for item in content)
            
            if isinstance(content, dict):
                # Convert dict to JSON-like string
                return str(content)
            
            # Convert other types to string
            return str(content)
            
        except Exception as e:
            logger.warning(f"Error validating content data: {e}")
            return str(content) if content is not None else ""


# Global validator instance
input_validator = InputValidator()


def safe_env_bool(var_name: str, default: bool = False) -> bool:
    """Convenience function for boolean environment variables"""
    return input_validator.safe_env_var(var_name, str(default), "boolean")


def safe_env_int(var_name: str, default: int = 0) -> int:
    """Convenience function for integer environment variables"""
    return input_validator.safe_env_var(var_name, str(default), "integer")


def safe_env_str(var_name: str, default: str = "") -> str:
    """Convenience function for string environment variables"""
    return input_validator.safe_env_var(var_name, default, "string")

