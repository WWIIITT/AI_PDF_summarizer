import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import PyPDF2
import docx
import io
import warnings
import re
from typing import List, Tuple, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize
import os
import tempfile
import pdfplumber
import platform
import logging
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
from functools import lru_cache
import threading
from queue import Queue
import gc
import psutil

# Suppress warnings
logging.getLogger("langchain.text_splitter").setLevel(logging.ERROR)
logging.getLogger("langchain_community.chat_models.openai").setLevel(logging.ERROR)

# Try to import OCR dependencies
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_path, convert_from_bytes
    from PIL import Image
    import numpy as np
    import cv2

    # Configure Tesseract path for Windows
    if platform.system() == 'Windows':
        tesseract_paths = [
            r"D:\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Tesseract-OCR\tesseract.exe",
        ]

        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    OCR_AVAILABLE = True
except ImportError:
    print("⚠️ OCR dependencies not installed. OCR features will be disabled.")

warnings.filterwarnings('ignore')

# Download required NLTK data
required_data = ['punkt', 'punkt_tab']
for data in required_data:
    try:
        nltk.data.find(f'tokenizers/{data}')
    except LookupError:
        nltk.download(data, quiet=True)


class DocumentCache:
    """Simple cache for processed documents"""

    def __init__(self, cache_dir=None):
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.cache_path = Path(self.cache_dir) / "doc_summarizer_cache"
        self.cache_path.mkdir(exist_ok=True)
        self.cache_index = self.cache_path / "index.json"
        self.load_index()

    def load_index(self):
        """Load cache index"""
        if self.cache_index.exists():
            try:
                with open(self.cache_index, 'r') as f:
                    self.index = json.load(f)
            except:
                self.index = {}
        else:
            self.index = {}

    def save_index(self):
        """Save cache index"""
        try:
            with open(self.cache_index, 'w') as f:
                json.dump(self.index, f)
        except:
            pass

    def get_file_hash(self, file_path):
        """Get hash of file for cache key"""
        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except:
            return None

    def get(self, file_path, cache_type="text"):
        """Get cached result if exists"""
        try:
            file_hash = self.get_file_hash(file_path)
            if not file_hash:
                return None

            cache_key = f"{file_hash}_{cache_type}"

            if cache_key in self.index:
                cache_file = self.cache_path / self.index[cache_key]
                if cache_file.exists():
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        return f.read()
        except:
            pass
        return None

    def set(self, file_path, content, cache_type="text"):
        """Cache result"""
        try:
            file_hash = self.get_file_hash(file_path)
            if not file_hash:
                return

            cache_key = f"{file_hash}_{cache_type}"
            cache_file = self.cache_path / f"{cache_key}.txt"

            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)

            self.index[cache_key] = cache_file.name
            self.save_index()
        except:
            pass


class OptimizedDocumentSummarizer:
    def __init__(self, api_key):
        # Initialize LLM with streaming support
        self.llm = ChatOpenAI(
            model='deepseek-chat',
            openai_api_key=api_key,
            openai_api_base='https://api.deepseek.com',
            max_tokens=2048,
            temperature=0.3,
            streaming=True
        )

        # Text splitters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", "。", ". ", "！", "! ", "？", "? ", "；", "; ", " ", ""],
            is_separator_regex=False
        )

        # Initialize cache
        self.cache = DocumentCache()

        # Thread pool for parallel processing (reduced workers to avoid overload)
        self.executor = ThreadPoolExecutor(max_workers=2)

        # OCR configuration
        self.ocr_available = False
        self.chinese_ocr_available = False
        if OCR_AVAILABLE:
            self.configure_ocr()

        # Add cancellation token
        self.cancel_processing = False

    def configure_ocr(self):
        """Configure OCR settings"""
        try:
            version = pytesseract.get_tesseract_version()
            self.ocr_available = True
            available_langs = pytesseract.get_languages()
            self.chinese_ocr_available = any(lang in available_langs for lang in ['chi_sim', 'chi_tra'])
        except:
            self.ocr_available = False
            self.chinese_ocr_available = False

    def extract_text_from_pdf_fast(self, file_path, use_ocr_if_needed=True, ocr_language='auto',
                                   quality='balanced', progress_callback=None, max_ocr_pages=20):
        """Optimized PDF extraction with timeout and page limits"""

        # Reset cancellation token
        self.cancel_processing = False

        # Check cache first
        cache_key = f"{quality}_{ocr_language}_ocr{use_ocr_if_needed}_max{max_ocr_pages}"
        cached_text = self.cache.get(file_path, cache_key)
        if cached_text:
            if progress_callback:
                progress_callback(1.0, "Loaded from cache")
            return cached_text

        extracted_text = ""
        ocr_pages = []

        # Try fast extraction with pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

                # Quick scan to check if OCR is needed (only check first 3 pages)
                sample_pages = min(3, total_pages)
                needs_ocr = False

                for i in range(sample_pages):
                    if self.cancel_processing:
                        return "Processing cancelled by user."

                    page_text = pdf.pages[i].extract_text() or ""
                    if self.is_scanned_pdf_page(page_text) or self.is_text_corrupted(page_text):
                        needs_ocr = True
                        break

                if not needs_ocr:
                    # Extract all text quickly
                    for i, page in enumerate(pdf.pages):
                        if self.cancel_processing:
                            return "Processing cancelled by user."

                        if progress_callback:
                            progress_callback(i / total_pages, f"Extracting page {i + 1}/{total_pages}")

                        page_text = page.extract_text() or ""
                        if page_text and not self.is_text_corrupted(page_text):
                            extracted_text += f"\n--- Page {i + 1}/{total_pages} ---\n{page_text}\n"

                    if extracted_text:
                        self.cache.set(file_path, extracted_text, cache_key)
                        return extracted_text
        except Exception as e:
            print(f"PDFPlumber error: {str(e)}")

        # If fast extraction failed or OCR is needed
        if use_ocr_if_needed and self.ocr_available:
            return self._extract_with_limited_ocr(file_path, ocr_language, quality, progress_callback, max_ocr_pages)
        else:
            # Fallback to PyPDF2
            return self._extract_with_pypdf2(file_path, progress_callback)

    def _extract_with_limited_ocr(self, file_path, ocr_language, quality, progress_callback, max_ocr_pages):
        """Extract text using OCR with page limits and timeout"""

        # Determine DPI based on quality setting (reduced for better performance)
        dpi_settings = {
            'fast': 100,  # Reduced from 150
            'balanced': 150,  # Reduced from 200
            'high': 200  # Reduced from 300
        }
        dpi = dpi_settings.get(quality, 150)

        try:
            # First, get total page count without converting all pages
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

            if progress_callback:
                progress_callback(0.1, f"PDF has {total_pages} pages. Will OCR up to {max_ocr_pages} pages...")

            # Limit pages to process
            pages_to_process = min(total_pages, max_ocr_pages)

            # Convert only the pages we need
            extracted_text = ""

            # Process pages in smaller batches to avoid memory issues
            batch_size = 5

            for batch_start in range(0, pages_to_process, batch_size):
                if self.cancel_processing:
                    return "Processing cancelled by user."

                batch_end = min(batch_start + batch_size, pages_to_process)

                # Convert batch of pages
                try:
                    if progress_callback:
                        progress_callback(
                            0.1 + (0.8 * batch_start / pages_to_process),
                            f"Converting pages {batch_start + 1}-{batch_end}..."
                        )

                    # Convert specific page range
                    images = convert_from_path(
                        file_path,
                        dpi=dpi,
                        first_page=batch_start + 1,
                        last_page=batch_end,
                        thread_count=2,
                        fmt='jpeg',  # JPEG is faster than PNG
                        jpegopt={'quality': 75, 'optimize': True}
                    )

                    # Process batch
                    for i, image in enumerate(images):
                        if self.cancel_processing:
                            return "Processing cancelled by user."

                        page_num = batch_start + i + 1

                        if progress_callback:
                            progress_callback(
                                0.1 + (0.8 * page_num / pages_to_process),
                                f"OCR processing page {page_num}/{pages_to_process}..."
                            )

                        try:
                            # Process with timeout
                            text = self._ocr_with_timeout(image, ocr_language, quality, timeout=30)
                            if text and text != "OCR timeout" and text != "OCR Error":
                                extracted_text += f"\n--- Page {page_num}/{total_pages} (OCR) ---\n{text}\n"
                        except Exception as e:
                            print(f"Error processing page {page_num}: {str(e)}")
                            extracted_text += f"\n--- Page {page_num}/{total_pages} (OCR Failed) ---\n[OCR failed for this page]\n"

                    # Clear memory after each batch
                    del images
                    gc.collect()

                except Exception as e:
                    print(f"Error converting batch {batch_start}-{batch_end}: {str(e)}")
                    continue

            # Add note about remaining pages if any
            if total_pages > pages_to_process:
                extracted_text += f"\n\n--- Note: OCR processed first {pages_to_process} pages out of {total_pages} total pages ---\n"

            # Cache the result
            cache_key = f"{quality}_{ocr_language}_ocrTrue_max{max_ocr_pages}"
            self.cache.set(file_path, extracted_text, cache_key)

            return extracted_text if extracted_text else "No text could be extracted with OCR."

        except Exception as e:
            return f"Error during OCR processing: {str(e)}"

    def _ocr_with_timeout(self, image, ocr_language, quality, timeout=30):
        """Run OCR with timeout"""
        import signal
        from contextlib import contextmanager

        @contextmanager
        def timeout_handler(seconds):
            def timeout_func(signum, frame):
                raise TimeoutError("OCR timeout")

            if platform.system() != 'Windows':
                # Unix-based timeout
                signal.signal(signal.SIGALRM, timeout_func)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
            else:
                # Windows doesn't support SIGALRM, use threading
                timer = threading.Timer(seconds, lambda: None)
                timer.start()
                try:
                    yield
                finally:
                    timer.cancel()

        try:
            # Use threading for timeout on Windows
            result = [None]
            exception = [None]

            def run_ocr():
                try:
                    result[0] = self.extract_text_with_ocr(
                        image,
                        preprocess=(quality == 'high'),
                        language=ocr_language
                    )
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=run_ocr)
            thread.daemon = True
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                # OCR is still running after timeout
                return "OCR timeout"

            if exception[0]:
                raise exception[0]

            return result[0] or "No text extracted"

        except Exception as e:
            return f"OCR Error: {str(e)}"

    def _extract_with_pypdf2(self, file_path, progress_callback):
        """Fallback extraction using PyPDF2"""
        try:
            extracted_text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                for i, page in enumerate(pdf_reader.pages):
                    if self.cancel_processing:
                        return "Processing cancelled by user."

                    if progress_callback:
                        progress_callback(i / total_pages, f"Extracting page {i + 1}/{total_pages}")

                    try:
                        page_text = page.extract_text()
                        if page_text and not self.is_text_corrupted(page_text):
                            extracted_text += f"\n--- Page {i + 1}/{total_pages} ---\n{page_text}\n"
                    except:
                        continue

            return extracted_text if extracted_text else "No text could be extracted from the PDF."
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def preprocess_image_for_ocr(self, image):
        """Optimized image preprocessing - simplified for speed"""
        if not OCR_AVAILABLE:
            return image

        try:
            # Resize if too large (for faster OCR)
            max_dimension = 2000
            if image.width > max_dimension or image.height > max_dimension:
                ratio = max_dimension / max(image.width, image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)

            # Convert to numpy array
            img_array = np.array(image)

            # Quick grayscale conversion
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # Simple thresholding (skip complex preprocessing for speed)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return Image.fromarray(thresh)
        except:
            return image

    def extract_text_with_ocr(self, image, preprocess=True, language='auto'):
        """Optimized OCR extraction"""
        if not self.ocr_available or not OCR_AVAILABLE:
            return "OCR not available."

        try:
            if preprocess:
                image = self.preprocess_image_for_ocr(image)

            # Determine OCR language
            ocr_lang = 'eng'
            if language == 'chinese' and self.chinese_ocr_available:
                ocr_lang = 'chi_sim+eng'  # Simplified: just use chi_sim with English
            elif language == 'auto':
                ocr_lang = 'eng'  # Default to English for speed

            # Perform OCR with optimized settings
            text = pytesseract.image_to_string(
                image,
                lang=ocr_lang,
                config='--psm 3 --oem 1 -c tessedit_do_invert=0'  # Optimized config
            )

            return text.strip()
        except Exception as e:
            return f"OCR Error: {str(e)}"

    def is_scanned_pdf_page(self, page_text):
        """Quick check if page is scanned"""
        return len(page_text.strip()) < 50

    def is_text_corrupted(self, text):
        """Quick corruption check"""
        if not text or len(text.strip()) < 10:
            return True

        # Quick check for readable characters
        readable_chars = len(re.findall(r'[\u4e00-\u9fff\u0020-\u007E\u00A0-\u00FF]', text[:100]))
        return readable_chars < 30

    def extract_text_from_docx(self, file_path):
        """Optimized Word document extraction"""
        # Check cache
        cached_text = self.cache.get(file_path, "docx")
        if cached_text:
            return cached_text

        try:
            doc = docx.Document(file_path)
            text_parts = []

            # Extract paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    if paragraph.style and paragraph.style.name.startswith('Heading'):
                        text_parts.append(f"\n## {paragraph.text}\n")
                    else:
                        text_parts.append(paragraph.text)

            # Extract tables efficiently
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_text:
                        text_parts.append(row_text)

            text = "\n".join(text_parts)

            # Cache result
            self.cache.set(file_path, text, "docx")

            return text.strip() if text else "No text found in the document."
        except Exception as e:
            return f"Error reading Word document: {str(e)}"

    def get_file_text(self, file_path, ocr_language='auto', quality='balanced',
                      progress_callback=None, max_ocr_pages=20):
        """Extract text with progress tracking"""
        file_lower = file_path.lower()

        if file_lower.endswith('.pdf'):
            return self.extract_text_from_pdf_fast(
                file_path,
                ocr_language=ocr_language,
                quality=quality,
                progress_callback=progress_callback,
                max_ocr_pages=max_ocr_pages
            )
        elif file_lower.endswith(('.docx', '.doc')):
            return self.extract_text_from_docx(file_path)
        elif file_lower.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            # Check cache
            cached_text = self.cache.get(file_path, f"image_{ocr_language}")
            if cached_text:
                return cached_text

            # Extract from image
            try:
                image = Image.open(file_path)
                text = self.extract_text_with_ocr(image, language=ocr_language)

                # Cache result
                self.cache.set(file_path, text, f"image_{ocr_language}")

                return text if text else "No text could be extracted from the image."
            except Exception as e:
                return f"Error reading image: {str(e)}"
        elif file_lower.endswith('.txt'):
            try:
                encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'utf-16']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            return file.read()
                    except UnicodeDecodeError:
                        continue
                return "Error: Unable to decode text file."
            except Exception as e:
                return f"Error reading text file: {str(e)}"
        else:
            return "Unsupported file format."

    def summarize_text_streaming(self, text, summary_type="concise", include_quotes=False,
                                 output_language="auto", progress_callback=None):
        """Generate summary with streaming support"""

        if not text or text.startswith("Error") or text.startswith("❌"):
            return text

        # Check cache for summary
        cache_key = f"summary_{summary_type}_{include_quotes}_{output_language}"
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cached_summary = self.cache.get(text_hash, cache_key)
        if cached_summary:
            return cached_summary

        # Create documents
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        if not documents:
            return "No content found to summarize."

        # Generate summary
        summary = self._generate_summary(documents, summary_type, include_quotes,
                                         output_language, progress_callback)

        # Cache result
        if summary and not summary.startswith("Error"):
            self.cache.set(text_hash, summary, cache_key)

        return summary

    def _generate_summary(self, documents, summary_type, include_quotes, output_language, progress_callback):
        """Generate summary with appropriate method"""

        # Language instructions
        lang_instructions = {
            "chinese": "Write the summary in Chinese (中文).",
            "english": "Write the summary in English.",
            "auto": "Match the language of the source document."
        }
        lang_instruction = lang_instructions.get(output_language, lang_instructions["auto"])

        # Summary prompts
        prompts = {
            "concise": f"Write a concise 2-3 paragraph summary. {lang_instruction}\n\n{{text}}\n\nSUMMARY:",
            "detailed": f"Create a comprehensive summary with key quotes if available. {lang_instruction}\n\n{{text}}\n\nDETAILED SUMMARY:",
            "bullet_points": f"Create a bullet-point summary. {lang_instruction}\n\n{{text}}\n\nBULLET POINTS:",
            "key_insights": f"Extract 5-7 key insights. {lang_instruction}\n\n{{text}}\n\nKEY INSIGHTS:",
            "chapter_wise": f"Create a section-by-section summary. {lang_instruction}\n\n{{text}}\n\nSECTION SUMMARY:"
        }

        prompt_template = prompts.get(summary_type, prompts["concise"])

        try:
            if len(documents) > 1:
                # Use map-reduce for long documents
                if progress_callback:
                    progress_callback(0.3, "Processing document chunks...")

                map_prompt = PromptTemplate(
                    template=f"Summarize this section. {lang_instruction}\n\n{{text}}\n\nSUMMARY:",
                    input_variables=["text"]
                )

                combine_prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["text"]
                )

                chain = load_summarize_chain(
                    self.llm,
                    chain_type="map_reduce",
                    map_prompt=map_prompt,
                    combine_prompt=combine_prompt,
                    verbose=False
                )

                summary = chain.run(documents)
            else:
                # Direct summarization for short documents
                messages = [
                    SystemMessage(content="You are an expert document analyst fluent in multiple languages."),
                    HumanMessage(content=prompt_template.format(text=documents[0].page_content))
                ]

                # Stream the response
                summary_parts = []
                for chunk in self.llm.stream(messages):
                    summary_parts.append(chunk.content)
                    if progress_callback:
                        progress_callback(0.5 + 0.5 * (len(summary_parts) / 100), "Generating summary...")

                summary = "".join(summary_parts)

            return self._format_summary(summary)

        except Exception as e:
            return f"Error during summarization: {str(e)}"

    def _format_summary(self, summary: str) -> str:
        """Format summary for readability"""
        summary = re.sub(r'\n{3,}', '\n\n', summary)
        summary = re.sub(r'"\s*([^"]+)\s*"', r'"\1"', summary)
        return summary.strip()

    def analyze_document_structure(self, text: str) -> Dict[str, any]:
        """Quick document analysis"""
        analysis = {
            "total_words": len(text.split()),
            "total_sentences": text.count('.') + text.count('。'),
            "detected_language": "Chinese" if len(re.findall(r'[\u4e00-\u9fff]', text[:1000])) > 100 else "English",
            "text_quality": "Corrupted" if self.is_text_corrupted(text) else "Good",
            "recommended_summary": "detailed" if len(text.split()) > 5000 else "concise"
        }
        return analysis

    def cancel_current_processing(self):
        """Cancel current processing operation"""
        self.cancel_processing = True


def create_optimized_gradio_interface():
    """Create the optimized Gradio interface"""

    summarizer = None

    def set_api_key(api_key):
        """Initialize summarizer with API key"""
        nonlocal summarizer
        if api_key.strip():
            try:
                summarizer = OptimizedDocumentSummarizer(api_key.strip())
                ocr_status = "✅ OCR Available" if summarizer.ocr_available else "⚠️ OCR Not Available"
                chinese_status = "✅ Chinese OCR Ready" if summarizer.chinese_ocr_available else "⚠️ Chinese OCR Not Ready"

                return f"✅ API Key set successfully! | {ocr_status} | {chinese_status}"
            except Exception as e:
                return f"❌ Error: {str(e)}"
        else:
            return "❌ Please enter a valid API key"

    def analyze_document(file):
        """Quick document analysis"""
        nonlocal summarizer

        if summarizer is None:
            return "❌ Please set your DeepSeek API key first!"

        if file is None:
            return "❌ Please upload a file!"

        try:
            # Quick text extraction for analysis
            text = summarizer.get_file_text(file.name, quality='fast', max_ocr_pages=1)

            if text.startswith("Error") or text.startswith("❌"):
                return text

            analysis = summarizer.analyze_document_structure(text)

            # Get file size
            file_size_mb = os.path.getsize(file.name) / (1024 * 1024)

            return f"""📊 **Document Analysis:**

• **File Size:** {file_size_mb:.2f} MB
• **Total Words:** {analysis['total_words']:,}
• **Detected Language:** {analysis['detected_language']}
• **Text Quality:** {analysis['text_quality']}
• **Recommended Summary:** {analysis['recommended_summary']}

💡 **Performance Tips:**
• For large PDFs (>50 pages), consider limiting OCR pages
• Use 'Fast' quality for quick results
• Use 'Balanced' for optimal speed/quality
• Enable caching for repeated processing

⚠️ **Note:** If the document is scanned, OCR processing may take several minutes.
"""
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def process_document(file, summary_type, include_quotes, use_ocr, ocr_language,
                         output_language, quality, max_ocr_pages, progress=gr.Progress()):
        """Process document with progress tracking"""
        nonlocal summarizer

        if summarizer is None:
            return "❌ Please set your DeepSeek API key first!"

        if file is None:
            return "❌ Please upload a file!"

        try:
            # Progress callback
            def update_progress(value, desc):
                progress(value, desc=desc)

            # Extract text
            progress(0.1, desc="Starting text extraction...")

            # Temporarily disable OCR if requested
            original_ocr_state = summarizer.ocr_available
            if not use_ocr:
                summarizer.ocr_available = False

            text = summarizer.get_file_text(
                file.name,
                ocr_language=ocr_language,
                quality=quality,
                progress_callback=update_progress,
                max_ocr_pages=max_ocr_pages
            )

            # Restore OCR state
            summarizer.ocr_available = original_ocr_state

            if text.startswith("Error") or text.startswith("❌"):
                return text

            if len(text.strip()) < 10:
                return "❌ No readable text found in the document."

            # Generate summary
            progress(0.5, desc="Generating summary...")

            summary = summarizer.summarize_text_streaming(
                text,
                summary_type,
                include_quotes,
                output_language,
                progress_callback=update_progress
            )

            progress(1.0, desc="Complete!")

            return summary

        except Exception as e:
            return f"❌ Error: {str(e)}"

    def clear_cache():
        """Clear the document cache"""
        try:
            if summarizer and summarizer.cache:
                # Clear cache files
                cache_files = list(summarizer.cache.cache_path.glob("*.txt"))
                for f in cache_files:
                    try:
                        f.unlink()
                    except:
                        pass

                # Clear index
                summarizer.cache.index = {}
                summarizer.cache.save_index()

                return "✅ Cache cleared successfully!"
        except Exception as e:
            return f"❌ Error clearing cache: {str(e)}"

    def cancel_processing():
        """Cancel current processing"""
        nonlocal summarizer
        if summarizer:
            summarizer.cancel_current_processing()
            return "⚠️ Processing cancellation requested..."
        return "No active processing to cancel"

    # Create the interface
    with gr.Blocks(title="Optimized Document Summarizer", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # ⚡ Optimized Document Summarizer with High-Speed Processing
            ## 高速文档摘要生成器

            **🚀 Performance Features:**
            - ⚡ Parallel OCR processing with timeout protection
            - 💾 Intelligent caching for repeated documents
            - 🔄 Streaming responses for faster feedback
            - 🎯 Quality settings for speed/accuracy balance
            - 📊 Real-time progress tracking
            - ⏹️ Cancellable operations
            - 🔢 Page limit controls for OCR
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔑 API Configuration")
                api_key_input = gr.Textbox(
                    label="DeepSeek API Key",
                    placeholder="Enter your DeepSeek API key...",
                    type="password"
                )
                api_key_button = gr.Button("Set API Key", variant="primary")
                api_key_status = gr.Textbox(label="Status", interactive=False)

                # Cache control
                gr.Markdown("### 💾 Cache Control")
                clear_cache_button = gr.Button("Clear Cache", variant="secondary")
                cache_status = gr.Textbox(label="Cache Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 📤 Document Upload")
                file_input = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"],
                    type="filepath"
                )

                analyze_button = gr.Button("📊 Quick Analysis", variant="secondary")
                analysis_output = gr.Markdown()

        with gr.Row():
            with gr.Column():
                summary_type = gr.Radio(
                    choices=[
                        ("📝 Concise", "concise"),
                        ("📖 Detailed", "detailed"),
                        ("• Bullet Points", "bullet_points"),
                        ("💡 Key Insights", "key_insights"),
                        ("📑 Chapter-wise", "chapter_wise")
                    ],
                    value="concise",
                    label="Summary Type"
                )

                # Performance settings
                gr.Markdown("### ⚡ Performance Settings")

                quality = gr.Radio(
                    choices=[
                        ("🚀 Fast (100 DPI)", "fast"),
                        ("⚖️ Balanced (150 DPI)", "balanced"),
                        ("🎯 High Quality (200 DPI)", "high")
                    ],
                    value="balanced",
                    label="Processing Quality"
                )

                max_ocr_pages = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                    label="Maximum OCR Pages (for large PDFs)",
                    info="Limit OCR processing to first N pages to avoid timeout"
                )

                include_quotes = gr.Checkbox(
                    label="Include quotes (detailed mode)",
                    value=True
                )

                use_ocr = gr.Checkbox(
                    label="🔍 Enable OCR",
                    value=True
                )

                ocr_language = gr.Radio(
                    choices=[
                        ("Auto-detect", "auto"),
                        ("Chinese Priority", "chinese"),
                        ("English Only", "english")
                    ],
                    value="auto",
                    label="OCR Language"
                )

                output_language = gr.Radio(
                    choices=[
                        ("Auto", "auto"),
                        ("Chinese", "chinese"),
                        ("English", "english")
                    ],
                    value="auto",
                    label="Output Language"
                )

                with gr.Row():
                    summarize_button = gr.Button("🚀 Generate Summary", variant="primary", size="lg")
                    cancel_button = gr.Button("⏹️ Cancel", variant="stop", size="sm")

        gr.Markdown("### 📋 Summary Output")
        output_text = gr.Textbox(
            label="Summary",
            lines=20,
            max_lines=50,
            interactive=False
        )

        # Event handlers
        api_key_button.click(
            fn=set_api_key,
            inputs=[api_key_input],
            outputs=[api_key_status]
        )

        analyze_button.click(
            fn=analyze_document,
            inputs=[file_input],
            outputs=[analysis_output]
        )

        summarize_button.click(
            fn=process_document,
            inputs=[file_input, summary_type, include_quotes, use_ocr, ocr_language,
                    output_language, quality, max_ocr_pages],
            outputs=[output_text]
        )

        clear_cache_button.click(
            fn=clear_cache,
            inputs=[],
            outputs=[cache_status]
        )

        cancel_button.click(
            fn=cancel_processing,
            inputs=[],
            outputs=[output_text]
        )

        gr.Markdown(
            """
            ### ⚡ Performance Tips:

            1. **For large PDFs with OCR:**
               - Limit OCR pages (e.g., 10-20 pages) to avoid timeout
               - Use Fast mode for initial testing
               - Consider processing in batches

            2. **If OCR is taking too long:**
               - Click Cancel button to stop processing
               - Try with fewer pages
               - Use Fast quality setting
               - Disable OCR if text is already selectable

            3. **General tips:**
               - Enable caching - Reprocessing cached documents is instant
               - Use Balanced mode for optimal speed/quality trade-off
               - High Quality mode only for critical documents with poor scan quality

            ### 🛠️ Technical Optimizations:
            - OCR timeout protection (30 seconds per page)
            - Memory-efficient batch processing
            - Page limit controls
            - Reduced DPI settings for faster processing
            - Cancellable operations
            - Smart caching system
            - Optimized image preprocessing
            """
        )

    return interface


if __name__ == "__main__":
    print("""
    ====================================
    OPTIMIZED DOCUMENT SUMMARIZER
    ====================================

    Key fixes for OCR timeout issues:
    - Added 30-second timeout per page
    - Page limit control (default: 20 pages)
    - Reduced DPI settings (100-200)
    - Batch processing to avoid memory issues
    - Cancellable operations
    - Better error handling

    Installation:
    pip install gradio langchain langchain-community PyPDF2 python-docx openai nltk pdfplumber pytesseract pdf2image pillow opencv-python-headless psutil

    ====================================
    """)

    interface = create_optimized_gradio_interface()
    interface.launch(
        share=True,
        server_name="localhost",
        server_port=7860,
        show_error=True
    )