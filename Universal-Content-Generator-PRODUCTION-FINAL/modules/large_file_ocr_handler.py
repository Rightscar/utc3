"""
Large File OCR Handler with Timeout Protection
Handles 100+ page scanned PDFs without breaking on Render.com deployment
"""

import os
import time
import logging
import tempfile
import threading
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass
import streamlit as st

# OCR and PDF processing
try:
    import pytesseract
    import pdf2image
    from PyPDF2 import PdfReader
    TESSERACT_AVAILABLE = True
    PDFREADER_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    PDFREADER_AVAILABLE = False
    # Create dummy classes for when imports fail
    class PdfReader:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyPDF2 not available")
        
        @property
        def pages(self):
            return []

# PIL Image import (needed for type hints)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    # Create a dummy Image class for type hints when PIL is not available
    class Image:
        class Image:
            pass

@dataclass
class OCRConfig:
    """Configuration for OCR processing"""
    max_file_size_mb: int = 100
    max_pages: int = 200
    timeout_per_page: int = 30  # seconds
    total_timeout: int = 1800   # 30 minutes max
    cpu_check_interval: int = 10  # seconds
    max_cpu_time: int = 600     # 10 minutes CPU time
    dpi: int = 150              # Lower DPI for faster processing
    tesseract_config: str = '--psm 6 --oem 3'  # Optimized config

class LargeFileOCRHandler:
    """Handles large file OCR with timeout and CPU protection"""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(__name__)
        self.start_time = None
        self.cpu_start_time = None
        self.processed_pages = 0
        self.skipped_pages = 0
        self.total_pages = 0
        self.is_cancelled = False
        
    def validate_file_for_ocr(self, file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate if file is suitable for OCR processing
        
        Returns:
            (is_valid, message, metadata)
        """
        try:
            # Check file size
            file_size = os.path.getsize(file_path)
            file_size_mb = file_size / (1024 * 1024)
            
            if file_size_mb > self.config.max_file_size_mb:
                return False, f"File too large ({file_size_mb:.1f}MB). Max: {self.config.max_file_size_mb}MB", {}
            
            # Check if it's a PDF
            if not file_path.lower().endswith('.pdf'):
                return False, "Only PDF files supported for OCR", {}
            
            # Check if PyPDF2 is available
            if not PDFREADER_AVAILABLE:
                return False, "PyPDF2 not available - cannot process PDF files", {}
            
            # Try to read PDF and count pages
            try:
                reader = PdfReader(file_path)
                page_count = len(reader.pages)
                
                if page_count > self.config.max_pages:
                    return False, f"Too many pages ({page_count}). Max: {self.config.max_pages}", {}
                
                # Check if PDF has extractable text (not image-only)
                text_pages = 0
                image_pages = 0
                
                for i, page in enumerate(reader.pages[:min(5, page_count)]):  # Sample first 5 pages
                    try:
                        text = page.extract_text().strip()
                        if len(text) > 50:  # Has meaningful text
                            text_pages += 1
                        else:
                            image_pages += 1
                    except:
                        image_pages += 1
                
                needs_ocr = image_pages > text_pages
                
                metadata = {
                    'file_size_mb': file_size_mb,
                    'page_count': page_count,
                    'needs_ocr': needs_ocr,
                    'estimated_time_minutes': self._estimate_processing_time(page_count, needs_ocr),
                    'text_pages': text_pages,
                    'image_pages': image_pages
                }
                
                return True, "File validated successfully", metadata
                
            except Exception as e:
                return False, f"Cannot read PDF: {str(e)}", {}
                
        except Exception as e:
            self.logger.error(f"File validation error: {e}")
            return False, f"Validation error: {str(e)}", {}
    
    def _estimate_processing_time(self, page_count: int, needs_ocr: bool) -> float:
        """Estimate processing time in minutes"""
        if not needs_ocr:
            return page_count * 0.1  # 0.1 minutes per text page
        else:
            return page_count * 2.0  # 2 minutes per OCR page
    
    def process_large_pdf_with_timeout(
        self, 
        file_path: str, 
        progress_callback: Optional[callable] = None,
        user_choice_callback: Optional[callable] = None
    ) -> Tuple[str, bool, Dict[str, Any]]:
        """
        Process large PDF with comprehensive timeout protection
        
        Args:
            file_path: Path to PDF file
            progress_callback: Function to call with progress updates
            user_choice_callback: Function to ask user about problematic pages
            
        Returns:
            (extracted_text, success, metadata)
        """
        self.start_time = time.time()
        self.cpu_start_time = time.process_time()
        self.processed_pages = 0
        self.skipped_pages = 0
        self.is_cancelled = False
        
        # Validate file first
        is_valid, message, file_metadata = self.validate_file_for_ocr(file_path)
        if not is_valid:
            return "", False, {'error': message, **file_metadata}
        
        self.total_pages = file_metadata['page_count']
        needs_ocr = file_metadata['needs_ocr']
        
        try:
            if needs_ocr and not TESSERACT_AVAILABLE:
                return "", False, {'error': 'Tesseract OCR not available. Please install tesseract-ocr.'}
            
            # Process PDF
            if needs_ocr:
                text, success, metadata = self._process_with_ocr(file_path, progress_callback, user_choice_callback)
            else:
                text, success, metadata = self._process_text_pdf(file_path, progress_callback)
            
            # Add processing statistics
            processing_time = time.time() - self.start_time
            cpu_time = time.process_time() - self.cpu_start_time
            
            metadata.update({
                'processing_time_seconds': processing_time,
                'cpu_time_seconds': cpu_time,
                'processed_pages': self.processed_pages,
                'skipped_pages': self.skipped_pages,
                'total_pages': self.total_pages,
                'success_rate': self.processed_pages / self.total_pages if self.total_pages > 0 else 0,
                **file_metadata
            })
            
            return text, success, metadata
            
        except Exception as e:
            self.logger.error(f"PDF processing error: {e}")
            return "", False, {'error': f"Processing failed: {str(e)}"}
    
    def _process_text_pdf(self, file_path: str, progress_callback: Optional[callable] = None) -> Tuple[str, bool, Dict]:
        """Process PDF with extractable text"""
        try:
            # Check if PyPDF2 is available
            if not PDFREADER_AVAILABLE:
                return "", False, {'error': "PyPDF2 not available - cannot process PDF files"}
            
            reader = PdfReader(file_path)
            extracted_text = []
            
            for i, page in enumerate(reader.pages):
                # Check timeouts
                if self._check_timeouts():
                    break
                
                try:
                    text = page.extract_text()
                    if text.strip():
                        extracted_text.append(text)
                        self.processed_pages += 1
                    else:
                        self.skipped_pages += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error extracting page {i+1}: {e}")
                    self.skipped_pages += 1
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(reader.pages)
                    progress_callback(progress, f"Processing page {i+1}/{len(reader.pages)}")
            
            full_text = "\n\n".join(extracted_text)
            success = len(full_text.strip()) > 0
            
            return full_text, success, {'method': 'text_extraction'}
            
        except Exception as e:
            self.logger.error(f"Text PDF processing error: {e}")
            return "", False, {'error': str(e)}
    
    def _process_with_ocr(
        self, 
        file_path: str, 
        progress_callback: Optional[callable] = None,
        user_choice_callback: Optional[callable] = None
    ) -> Tuple[str, bool, Dict]:
        """Process PDF with OCR"""
        try:
            # Convert PDF to images with timeout protection
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Convert with lower DPI for speed
                    images = pdf2image.convert_from_path(
                        file_path,
                        dpi=self.config.dpi,
                        output_folder=temp_dir,
                        fmt='jpeg',
                        thread_count=1  # Single thread to control CPU usage
                    )
                except Exception as e:
                    return "", False, {'error': f"PDF to image conversion failed: {str(e)}"}
                
                extracted_text = []
                failed_pages = []
                
                for i, image in enumerate(images):
                    # Check timeouts
                    if self._check_timeouts():
                        break
                    
                    page_num = i + 1
                    page_start_time = time.time()
                    
                    try:
                        # OCR with timeout per page
                        text = self._ocr_image_with_timeout(image, self.config.timeout_per_page)
                        
                        if text and len(text.strip()) > 10:  # Meaningful text
                            extracted_text.append(f"--- Page {page_num} ---\n{text}")
                            self.processed_pages += 1
                        else:
                            failed_pages.append(page_num)
                            self.skipped_pages += 1
                            
                    except Exception as e:
                        self.logger.warning(f"OCR failed for page {page_num}: {e}")
                        failed_pages.append(page_num)
                        self.skipped_pages += 1
                        
                        # Ask user if they want to continue
                        if user_choice_callback and len(failed_pages) > 5:
                            should_continue = user_choice_callback(
                                f"OCR failed for {len(failed_pages)} pages. Continue processing remaining pages?",
                                failed_pages
                            )
                            if not should_continue:
                                self.is_cancelled = True
                                break
                    
                    # Update progress
                    if progress_callback:
                        progress = (i + 1) / len(images)
                        elapsed = time.time() - page_start_time
                        progress_callback(
                            progress, 
                            f"OCR page {page_num}/{len(images)} ({elapsed:.1f}s)"
                        )
                
                full_text = "\n\n".join(extracted_text)
                success = len(full_text.strip()) > 0
                
                metadata = {
                    'method': 'ocr',
                    'failed_pages': failed_pages,
                    'ocr_config': self.config.tesseract_config,
                    'dpi': self.config.dpi
                }
                
                return full_text, success, metadata
                
        except Exception as e:
            self.logger.error(f"OCR processing error: {e}")
            return "", False, {'error': str(e)}
    
    def _ocr_image_with_timeout(self, image: Image.Image, timeout: int) -> str:
        """Perform OCR on image with timeout"""
        result = [None]
        exception = [None]
        
        def ocr_worker():
            try:
                result[0] = pytesseract.image_to_string(
                    image, 
                    config=self.config.tesseract_config
                )
            except Exception as e:
                exception[0] = e
        
        thread = threading.Thread(target=ocr_worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"OCR timeout after {timeout} seconds")
        
        if exception[0]:
            raise exception[0]
        
        return result[0] or ""
    
    def _check_timeouts(self) -> bool:
        """Check if any timeout conditions are met"""
        current_time = time.time()
        current_cpu_time = time.process_time()
        
        # Check total timeout
        if current_time - self.start_time > self.config.total_timeout:
            self.logger.warning("Total timeout exceeded")
            return True
        
        # Check CPU timeout
        if current_cpu_time - self.cpu_start_time > self.config.max_cpu_time:
            self.logger.warning("CPU timeout exceeded")
            return True
        
        # Check if cancelled by user
        if self.is_cancelled:
            self.logger.info("Processing cancelled by user")
            return True
        
        return False
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        if not self.start_time:
            return {'status': 'not_started'}
        
        current_time = time.time()
        current_cpu_time = time.process_time()
        
        elapsed_time = current_time - self.start_time
        cpu_time = current_cpu_time - self.cpu_start_time
        
        progress = self.processed_pages / self.total_pages if self.total_pages > 0 else 0
        
        return {
            'status': 'processing',
            'elapsed_time': elapsed_time,
            'cpu_time': cpu_time,
            'progress': progress,
            'processed_pages': self.processed_pages,
            'skipped_pages': self.skipped_pages,
            'total_pages': self.total_pages,
            'estimated_remaining': self._estimate_remaining_time()
        }
    
    def _estimate_remaining_time(self) -> float:
        """Estimate remaining processing time"""
        if self.processed_pages == 0:
            return 0
        
        elapsed = time.time() - self.start_time
        avg_time_per_page = elapsed / self.processed_pages
        remaining_pages = self.total_pages - self.processed_pages - self.skipped_pages
        
        return remaining_pages * avg_time_per_page

# Streamlit integration functions
def create_ocr_progress_ui():
    """Create progress UI for OCR processing"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_cols = st.columns(4)
    
    with metrics_cols[0]:
        processed_metric = st.empty()
    with metrics_cols[1]:
        skipped_metric = st.empty()
    with metrics_cols[2]:
        time_metric = st.empty()
    with metrics_cols[3]:
        cpu_metric = st.empty()
    
    return {
        'progress_bar': progress_bar,
        'status_text': status_text,
        'processed_metric': processed_metric,
        'skipped_metric': skipped_metric,
        'time_metric': time_metric,
        'cpu_metric': cpu_metric
    }

def update_ocr_progress_ui(ui_elements: Dict, handler: LargeFileOCRHandler, progress: float, message: str):
    """Update OCR progress UI"""
    ui_elements['progress_bar'].progress(progress)
    ui_elements['status_text'].text(message)
    
    status = handler.get_processing_status()
    
    ui_elements['processed_metric'].metric(
        "Processed Pages", 
        f"{status.get('processed_pages', 0)}/{status.get('total_pages', 0)}"
    )
    ui_elements['skipped_metric'].metric(
        "Skipped Pages", 
        status.get('skipped_pages', 0)
    )
    ui_elements['time_metric'].metric(
        "Elapsed Time", 
        f"{status.get('elapsed_time', 0):.1f}s"
    )
    ui_elements['cpu_metric'].metric(
        "CPU Time", 
        f"{status.get('cpu_time', 0):.1f}s"
    )

def show_ocr_file_warning(metadata: Dict[str, Any]) -> bool:
    """Show warning for large files and get user confirmation"""
    st.warning("⚠️ Large File Detected")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("File Size", f"{metadata.get('file_size_mb', 0):.1f} MB")
        st.metric("Pages", metadata.get('page_count', 0))
    
    with col2:
        st.metric("Estimated Time", f"{metadata.get('estimated_time_minutes', 0):.1f} min")
        st.metric("Needs OCR", "Yes" if metadata.get('needs_ocr', False) else "No")
    
    if metadata.get('needs_ocr', False):
        st.info("📄 This appears to be a scanned PDF that requires OCR processing. This may take several minutes.")
    
    return st.button("🚀 Process File", help="Start processing this large file")

# Example usage function
def process_large_file_with_ui(file_path: str) -> Tuple[str, bool, Dict]:
    """Process large file with full UI integration"""
    config = OCRConfig(
        max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', '100')),
        max_pages=int(os.getenv('MAX_PAGES', '200')),
        timeout_per_page=int(os.getenv('TIMEOUT_PER_PAGE', '30')),
        total_timeout=int(os.getenv('TOTAL_TIMEOUT', '1800'))
    )
    
    handler = LargeFileOCRHandler(config)
    
    # Validate file
    is_valid, message, metadata = handler.validate_file_for_ocr(file_path)
    
    if not is_valid:
        st.error(f"❌ {message}")
        return "", False, metadata
    
    # Show warning for large files
    if metadata.get('file_size_mb', 0) > 10 or metadata.get('page_count', 0) > 20:
        if not show_ocr_file_warning(metadata):
            return "", False, {'cancelled': True}
    
    # Create progress UI
    ui_elements = create_ocr_progress_ui()
    
    # Progress callback
    def progress_callback(progress: float, message: str):
        update_ocr_progress_ui(ui_elements, handler, progress, message)
    
    # User choice callback for failed pages
    def user_choice_callback(message: str, failed_pages: List[int]) -> bool:
        st.warning(f"⚠️ {message}")
        st.write(f"Failed pages: {', '.join(map(str, failed_pages[:10]))}")
        return st.button("Continue Processing", key=f"continue_{len(failed_pages)}")
    
    # Process file
    with st.spinner("Processing large file..."):
        text, success, result_metadata = handler.process_large_pdf_with_timeout(
            file_path, 
            progress_callback, 
            user_choice_callback
        )
    
    # Show results
    if success:
        st.success(f"✅ Processing completed! Extracted {len(text)} characters from {result_metadata.get('processed_pages', 0)} pages.")
        
        if result_metadata.get('skipped_pages', 0) > 0:
            st.warning(f"⚠️ Skipped {result_metadata.get('skipped_pages', 0)} pages due to errors.")
    else:
        st.error(f"❌ Processing failed: {result_metadata.get('error', 'Unknown error')}")
    
    return text, success, {**metadata, **result_metadata}

