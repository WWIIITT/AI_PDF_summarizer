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
import asyncio
from datetime import datetime

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
    print("⚠️ 未安装OCR依赖项。OCR功能将被禁用。")

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


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
        # Initialize LLM with timeout and reduced token limits
        self.llm = ChatOpenAI(
            model='deepseek-chat',
            openai_api_key=api_key,
            openai_api_base='https://api.deepseek.com',
            max_tokens=1024,  # Reduced from 2048
            temperature=0.3,
            streaming=True,
            request_timeout=60  # 60 second timeout for API calls
        )

        # Text splitters with smaller chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Reduced from 4000
            chunk_overlap=200,  # Reduced from 400
            length_function=len,
            separators=["\n\n", "\n", "。", ". ", "！", "! ", "？", "? ", "；", "; ", " ", ""],
            is_separator_regex=False
        )

        # Initialize cache
        self.cache = DocumentCache()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=2)

        # OCR configuration
        self.ocr_available = False
        self.chinese_ocr_available = False
        if OCR_AVAILABLE:
            self.configure_ocr()

        # Add cancellation token
        self.cancel_processing = False

        # Maximum text length to process (characters)
        self.max_text_length = 150000  # ~150k characters max

        # Maximum chunks to process
        self.max_chunks = 20

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
                progress_callback(1.0, "从缓存加载 Loaded from cache")
            return cached_text

        extracted_text = ""
        ocr_pages = []

        # Try fast extraction with pdfplumber first
        try:
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

                # Limit pages for very large PDFs
                max_pages_to_extract = min(total_pages, 100)  # Max 100 pages

                if total_pages > max_pages_to_extract:
                    if progress_callback:
                        progress_callback(0.05,
                                          f"大型PDF检测到：仅处理前{max_pages_to_extract}页 Large PDF detected: Processing first {max_pages_to_extract} pages only")

                # Quick scan to check if OCR is needed (only check first 3 pages)
                sample_pages = min(3, total_pages)
                needs_ocr = False

                for i in range(sample_pages):
                    if self.cancel_processing:
                        return "用户已取消处理。Processing cancelled by user."

                    page_text = pdf.pages[i].extract_text() or ""
                    if self.is_scanned_pdf_page(page_text) or self.is_text_corrupted(page_text):
                        needs_ocr = True
                        break

                if not needs_ocr:
                    # Extract text with length limit
                    for i in range(max_pages_to_extract):
                        if self.cancel_processing:
                            return "用户已取消处理。Processing cancelled by user."

                        if len(extracted_text) > self.max_text_length:
                            extracted_text += f"\n\n--- 达到文本长度限制，停止提取 Text length limit reached, stopping extraction ---\n"
                            break

                        if progress_callback:
                            progress_callback(i / max_pages_to_extract,
                                              f"提取第 {i + 1}/{max_pages_to_extract} 页 Extracting page {i + 1}/{max_pages_to_extract}")

                        try:
                            page_text = pdf.pages[i].extract_text() or ""
                            if page_text and not self.is_text_corrupted(page_text):
                                extracted_text += f"\n--- 第 {i + 1}/{total_pages} 页 Page {i + 1}/{total_pages} ---\n{page_text}\n"
                        except Exception as e:
                            print(f"Error extracting page {i + 1}: {str(e)}")
                            continue

                    if extracted_text:
                        # Truncate if still too long
                        if len(extracted_text) > self.max_text_length:
                            extracted_text = extracted_text[:self.max_text_length] + "\n\n--- 文本已截断 Text truncated ---"

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

        # Determine DPI based on quality setting
        dpi_settings = {
            'fast': 100,
            'balanced': 150,
            'high': 200
        }
        dpi = dpi_settings.get(quality, 150)

        try:
            # First, get total page count
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

            if progress_callback:
                progress_callback(0.1,
                                  f"PDF共有 {total_pages} 页，将OCR处理前 {max_ocr_pages} 页... PDF has {total_pages} pages. Will OCR up to {max_ocr_pages} pages...")

            # Limit pages to process
            pages_to_process = min(total_pages, max_ocr_pages)

            # Convert only the pages we need
            extracted_text = ""

            # Process pages in smaller batches
            batch_size = 5

            for batch_start in range(0, pages_to_process, batch_size):
                if self.cancel_processing:
                    return "用户已取消处理。Processing cancelled by user."

                if len(extracted_text) > self.max_text_length:
                    extracted_text += f"\n\n--- 达到文本长度限制 Text length limit reached ---\n"
                    break

                batch_end = min(batch_start + batch_size, pages_to_process)

                # Convert batch of pages
                try:
                    if progress_callback:
                        progress_callback(
                            0.1 + (0.8 * batch_start / pages_to_process),
                            f"转换第 {batch_start + 1}-{batch_end} 页... Converting pages {batch_start + 1}-{batch_end}..."
                        )

                    # Convert specific page range
                    images = convert_from_path(
                        file_path,
                        dpi=dpi,
                        first_page=batch_start + 1,
                        last_page=batch_end,
                        thread_count=2,
                        fmt='jpeg',
                        jpegopt={'quality': 75, 'optimize': True}
                    )

                    # Process batch
                    for i, image in enumerate(images):
                        if self.cancel_processing:
                            return "用户已取消处理。Processing cancelled by user."

                        page_num = batch_start + i + 1

                        if progress_callback:
                            progress_callback(
                                0.1 + (0.8 * page_num / pages_to_process),
                                f"OCR处理第 {page_num}/{pages_to_process} 页... OCR processing page {page_num}/{pages_to_process}..."
                            )

                        try:
                            # Process with timeout
                            text = self._ocr_with_timeout(image, ocr_language, quality, timeout=30)
                            if text and text != "OCR timeout" and text != "OCR Error":
                                extracted_text += f"\n--- 第 {page_num}/{total_pages} 页 Page {page_num}/{total_pages} (OCR) ---\n{text}\n"
                        except Exception as e:
                            print(f"处理第 {page_num} 页时出错 Error processing page {page_num}: {str(e)}")
                            extracted_text += f"\n--- 第 {page_num}/{total_pages} 页 Page {page_num}/{total_pages} (OCR失败 Failed) ---\n[此页OCR失败 OCR failed for this page]\n"

                    # Clear memory after each batch
                    del images
                    gc.collect()

                except Exception as e:
                    print(
                        f"转换批次 {batch_start}-{batch_end} 时出错 Error converting batch {batch_start}-{batch_end}: {str(e)}")
                    continue

            # Add note about remaining pages if any
            if total_pages > pages_to_process:
                extracted_text += f"\n\n--- 注意：OCR仅处理了前 {pages_to_process} 页，共 {total_pages} 页 Note: OCR processed first {pages_to_process} pages out of {total_pages} total pages ---\n"

            # Truncate if too long
            if len(extracted_text) > self.max_text_length:
                extracted_text = extracted_text[:self.max_text_length] + "\n\n--- 文本已截断 Text truncated ---"

            # Cache the result
            cache_key = f"{quality}_{ocr_language}_ocrTrue_max{max_ocr_pages}"
            self.cache.set(file_path, extracted_text, cache_key)

            return extracted_text if extracted_text else "无法通过OCR提取文本。No text could be extracted with OCR."

        except Exception as e:
            return f"OCR处理时出错 Error during OCR processing: {str(e)}"

    def _ocr_with_timeout(self, image, ocr_language, quality, timeout=30):
        """Run OCR with timeout"""
        import signal
        from contextlib import contextmanager

        try:
            # Use threading for timeout
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
                return "OCR超时 OCR timeout"

            if exception[0]:
                raise exception[0]

            return result[0] or "未提取到文本 No text extracted"

        except Exception as e:
            return f"OCR错误 OCR Error: {str(e)}"

    def _extract_with_pypdf2(self, file_path, progress_callback):
        """Fallback extraction using PyPDF2"""
        try:
            extracted_text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                # Limit pages
                max_pages = min(total_pages, 100)

                for i in range(max_pages):
                    if self.cancel_processing:
                        return "用户已取消处理。Processing cancelled by user."

                    if len(extracted_text) > self.max_text_length:
                        extracted_text += f"\n\n--- 达到文本长度限制 Text length limit reached ---\n"
                        break

                    if progress_callback:
                        progress_callback(i / max_pages,
                                          f"提取第 {i + 1}/{max_pages} 页 Extracting page {i + 1}/{max_pages}")

                    try:
                        page_text = pdf_reader.pages[i].extract_text()
                        if page_text and not self.is_text_corrupted(page_text):
                            extracted_text += f"\n--- 第 {i + 1}/{total_pages} 页 Page {i + 1}/{total_pages} ---\n{page_text}\n"
                    except:
                        continue

            # Truncate if too long
            if len(extracted_text) > self.max_text_length:
                extracted_text = extracted_text[:self.max_text_length] + "\n\n--- 文本已截断 Text truncated ---"

            return extracted_text if extracted_text else "无法从PDF中提取文本。No text could be extracted from the PDF."
        except Exception as e:
            return f"读取PDF时出错 Error reading PDF: {str(e)}"

    def preprocess_image_for_ocr(self, image):
        """Optimized image preprocessing"""
        if not OCR_AVAILABLE:
            return image

        try:
            # Resize if too large
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

            # Simple thresholding
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            return Image.fromarray(thresh)
        except:
            return image

    def extract_text_with_ocr(self, image, preprocess=True, language='auto'):
        """Optimized OCR extraction"""
        if not self.ocr_available or not OCR_AVAILABLE:
            return "OCR不可用。OCR not available."

        try:
            if preprocess:
                image = self.preprocess_image_for_ocr(image)

            # Determine OCR language
            ocr_lang = 'eng'
            if language == 'chinese' and self.chinese_ocr_available:
                ocr_lang = 'chi_sim+eng'
            elif language == 'auto':
                ocr_lang = 'eng'

            # Perform OCR with optimized settings
            text = pytesseract.image_to_string(
                image,
                lang=ocr_lang,
                config='--psm 3 --oem 1 -c tessedit_do_invert=0'
            )

            return text.strip()
        except Exception as e:
            return f"OCR错误 OCR Error: {str(e)}"

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
                if len("".join(text_parts)) > self.max_text_length:
                    text_parts.append("\n\n--- 达到文本长度限制 Text length limit reached ---")
                    break

                if paragraph.text.strip():
                    if paragraph.style and paragraph.style.name.startswith('Heading'):
                        text_parts.append(f"\n## {paragraph.text}\n")
                    else:
                        text_parts.append(paragraph.text)

            # Extract tables efficiently
            for table in doc.tables:
                if len("".join(text_parts)) > self.max_text_length:
                    break

                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                    if row_text:
                        text_parts.append(row_text)

            text = "\n".join(text_parts)

            # Truncate if too long
            if len(text) > self.max_text_length:
                text = text[:self.max_text_length] + "\n\n--- 文本已截断 Text truncated ---"

            # Cache result
            self.cache.set(file_path, text, "docx")

            return text.strip() if text else "文档中未找到文本。No text found in the document."
        except Exception as e:
            return f"读取Word文档时出错 Error reading Word document: {str(e)}"

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

                return text if text else "无法从图片中提取文本。No text could be extracted from the image."
            except Exception as e:
                return f"读取图片时出错 Error reading image: {str(e)}"
        elif file_lower.endswith('.txt'):
            try:
                encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'utf-16']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            text = file.read()
                            # Limit text length
                            if len(text) > self.max_text_length:
                                text = text[:self.max_text_length] + "\n\n--- 文本已截断 Text truncated ---"
                            return text
                    except UnicodeDecodeError:
                        continue
                return "错误：无法解码文本文件。Error: Unable to decode text file."
            except Exception as e:
                return f"读取文本文件时出错 Error reading text file: {str(e)}"
        else:
            return "不支持的文件格式。Unsupported file format."

    def summarize_text_streaming(self, text, summary_type="concise", include_quotes=False,
                                 output_language="auto", progress_callback=None):
        """Generate summary with streaming support and timeout"""

        if not text or text.startswith("Error") or text.startswith("❌"):
            return text

        # Check cache for summary
        cache_key = f"summary_{summary_type}_{include_quotes}_{output_language}"
        text_hash = hashlib.md5(text.encode()).hexdigest()
        cached_summary = self.cache.get(text_hash, cache_key)
        if cached_summary:
            return cached_summary

        # Limit text length for summarization
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
            if progress_callback:
                progress_callback(0.2, "文本过长，已截断 Text too long, truncated")

        # Create documents with limited chunks
        chunks = self.text_splitter.split_text(text)

        # Limit number of chunks
        if len(chunks) > self.max_chunks:
            chunks = chunks[:self.max_chunks]
            if progress_callback:
                progress_callback(0.3,
                                  f"文档块过多，仅处理前{self.max_chunks}块 Too many chunks, processing first {self.max_chunks}")

        documents = [Document(page_content=chunk) for chunk in chunks]

        if not documents:
            return "未找到可总结的内容。No content found to summarize."

        # Generate summary with timeout protection
        try:
            summary = self._generate_summary_with_timeout(
                documents, summary_type, include_quotes,
                output_language, progress_callback,
                timeout=300  # 5 minute timeout for entire summarization
            )

            # Cache result
            if summary and not summary.startswith("Error") and not summary.startswith("超时"):
                self.cache.set(text_hash, summary, cache_key)

            return summary
        except Exception as e:
            return f"总结生成失败 Summarization failed: {str(e)}"

    def _generate_summary_with_timeout(self, documents, summary_type, include_quotes,
                                       output_language, progress_callback, timeout=300):
        """Generate summary with timeout protection"""

        result = [None]
        exception = [None]

        def run_summary():
            try:
                result[0] = self._generate_summary(
                    documents, summary_type, include_quotes,
                    output_language, progress_callback
                )
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=run_summary)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            self.cancel_processing = True
            return "超时：总结生成时间过长，请尝试减少文档大小或使用'简洁'模式 Timeout: Summary generation took too long. Try reducing document size or using 'concise' mode."

        if exception[0]:
            raise exception[0]

        return result[0] or "未能生成摘要 Failed to generate summary"

    def _generate_summary(self, documents, summary_type, include_quotes, output_language, progress_callback):
        """Generate summary with appropriate method"""

        # Language instructions
        lang_instructions = {
            "chinese": "用中文撰写摘要。Write the summary in Chinese (中文).",
            "english": "用英文撰写摘要。Write the summary in English.",
            "auto": "使用源文档的语言撰写摘要。Match the language of the source document."
        }
        lang_instruction = lang_instructions.get(output_language, lang_instructions["auto"])

        # Summary prompts (simplified for better performance)
        prompts = {
            "concise": f"简洁总结以下内容（2-3段）。Concisely summarize (2-3 paragraphs). {lang_instruction}\n\n{{text}}\n\n摘要 SUMMARY:",
            "detailed": f"详细总结以下内容。Provide detailed summary. {lang_instruction}\n\n{{text}}\n\n详细摘要 DETAILED SUMMARY:",
            "bullet_points": f"用要点总结。Summarize in bullet points. {lang_instruction}\n\n{{text}}\n\n要点 BULLET POINTS:",
            "key_insights": f"提取5个关键见解。Extract 5 key insights. {lang_instruction}\n\n{{text}}\n\n关键见解 KEY INSIGHTS:",
            "chapter_wise": f"按章节总结。Summarize by sections. {lang_instruction}\n\n{{text}}\n\n章节摘要 SECTION SUMMARY:"
        }

        prompt_template = prompts.get(summary_type, prompts["concise"])

        try:
            if len(documents) > 1:
                # Use simpler approach for multiple documents
                if progress_callback:
                    progress_callback(0.3,
                                      f"处理 {len(documents)} 个文档块... Processing {len(documents)} document chunks...")

                # Combine all documents first (faster than map-reduce for moderate sizes)
                if len(documents) <= 5:
                    combined_text = "\n\n".join([doc.page_content for doc in documents])

                    # Single API call for small documents
                    messages = [
                        SystemMessage(content="你是专业的文档分析师。You are a professional document analyst."),
                        HumanMessage(content=prompt_template.format(text=combined_text))
                    ]

                    summary = ""
                    chunk_count = 0

                    try:
                        for chunk in self.llm.stream(messages):
                            if self.cancel_processing:
                                return "用户已取消处理。Processing cancelled by user."

                            summary += chunk.content
                            chunk_count += 1

                            if progress_callback and chunk_count % 10 == 0:
                                progress_callback(0.5 + 0.4 * min(chunk_count / 100, 1),
                                                  "生成摘要中... Generating summary...")
                    except Exception as e:
                        return f"API调用失败 API call failed: {str(e)}"

                else:
                    # For larger documents, process in batches
                    batch_summaries = []
                    batch_size = 3

                    for i in range(0, len(documents), batch_size):
                        if self.cancel_processing:
                            return "用户已取消处理。Processing cancelled by user."

                        batch = documents[i:i + batch_size]
                        batch_text = "\n\n".join([doc.page_content for doc in batch])

                        if progress_callback:
                            progress_callback(0.3 + 0.4 * (i / len(documents)),
                                              f"处理批次 {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}... Processing batch {i // batch_size + 1}/{(len(documents) + batch_size - 1) // batch_size}...")

                        messages = [
                            SystemMessage(content="总结这部分内容。Summarize this section."),
                            HumanMessage(content=f"{lang_instruction}\n\n{batch_text}\n\n摘要 SUMMARY:")
                        ]

                        try:
                            batch_summary = self.llm.invoke(messages).content
                            batch_summaries.append(batch_summary)
                        except Exception as e:
                            print(f"批次处理失败 Batch processing failed: {str(e)}")
                            continue

                    # Combine batch summaries
                    if batch_summaries:
                        combined_summaries = "\n\n".join(batch_summaries)
                        final_messages = [
                            SystemMessage(
                                content="合并以下摘要为最终摘要。Combine these summaries into a final summary."),
                            HumanMessage(content=prompt_template.format(text=combined_summaries))
                        ]

                        summary = self.llm.invoke(final_messages).content
                    else:
                        return "无法生成摘要 Failed to generate summary"

            else:
                # Direct summarization for single document
                messages = [
                    SystemMessage(content="你是专业的文档分析师。You are a professional document analyst."),
                    HumanMessage(content=prompt_template.format(text=documents[0].page_content))
                ]

                summary = ""
                chunk_count = 0

                try:
                    for chunk in self.llm.stream(messages):
                        if self.cancel_processing:
                            return "用户已取消处理。Processing cancelled by user."

                        summary += chunk.content
                        chunk_count += 1

                        if progress_callback and chunk_count % 10 == 0:
                            progress_callback(0.5 + 0.4 * min(chunk_count / 100, 1),
                                              "生成摘要中... Generating summary...")
                except Exception as e:
                    return f"API调用失败 API call failed: {str(e)}"

            return self._format_summary(summary)

        except Exception as e:
            return f"摘要生成时出错 Error during summarization: {str(e)}"

    def _format_summary(self, summary: str) -> str:
        """Format summary for readability"""
        summary = re.sub(r'\n{3,}', '\n\n', summary)
        summary = re.sub(r'"\s*([^"]+)\s*"', r'"\1"', summary)
        return summary.strip()

    def analyze_document_structure(self, text: str) -> Dict[str, any]:
        """Quick document analysis"""
        analysis = {
            "total_words": len(text.split()),
            "total_characters": len(text),
            "total_sentences": text.count('.') + text.count('。'),
            "detected_language": "中文 Chinese" if len(
                re.findall(r'[\u4e00-\u9fff]', text[:1000])) > 100 else "英文 English",
            "text_quality": "损坏 Corrupted" if self.is_text_corrupted(text) else "良好 Good",
            "recommended_summary": "详细 detailed" if len(text.split()) > 5000 else "简洁 concise",
            "estimated_time": f"{max(1, len(text) // 10000)} 分钟 minutes"
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
                ocr_status = "✅ OCR可用 OCR Available" if summarizer.ocr_available else "⚠️ OCR不可用 OCR Not Available"
                chinese_status = "✅ 中文OCR就绪 Chinese OCR Ready" if summarizer.chinese_ocr_available else "⚠️ 中文OCR未就绪 Chinese OCR Not Ready"

                return f"✅ API密钥设置成功！API Key set successfully! | {ocr_status} | {chinese_status}"
            except Exception as e:
                return f"❌ 错误 Error: {str(e)}"
        else:
            return "❌ 请输入有效的API密钥 Please enter a valid API key"

    def analyze_document(file):
        """Quick document analysis"""
        nonlocal summarizer

        if summarizer is None:
            return "❌ 请先设置您的DeepSeek API密钥！Please set your DeepSeek API key first!"

        if file is None:
            return "❌ 请上传文件！Please upload a file!"

        try:
            # Quick text extraction for analysis
            text = summarizer.get_file_text(file.name, quality='fast', max_ocr_pages=1)

            if text.startswith("Error") or text.startswith("❌"):
                return text

            analysis = summarizer.analyze_document_structure(text)

            # Get file size
            file_size_mb = os.path.getsize(file.name) / (1024 * 1024)

            return f"""📊 **文档分析 Document Analysis:**

• **文件大小 File Size:** {file_size_mb:.2f} MB
• **总词数 Total Words:** {analysis['total_words']:,}
• **总字符 Total Characters:** {analysis['total_characters']:,}
• **检测语言 Detected Language:** {analysis['detected_language']}
• **文本质量 Text Quality:** {analysis['text_quality']}
• **推荐摘要类型 Recommended Summary:** {analysis['recommended_summary']}
• **预计处理时间 Estimated Time:** {analysis['estimated_time']}

⚠️ **重要限制 Important Limits:**
• 最大文本长度 Max text length: 100,000 字符 characters
• 最大文档块 Max chunks: 20
• API超时 API timeout: 60 秒 seconds
• 总处理超时 Total timeout: 5 分钟 minutes

💡 **性能提示 Performance Tips:**
• 大文档将自动截断 Large documents will be automatically truncated
• 使用"简洁"模式更快 Use 'Concise' mode for faster results
• 禁用OCR如果不需要 Disable OCR if not needed
• 考虑分割大文档 Consider splitting large documents
"""
        except Exception as e:
            return f"❌ 错误 Error: {str(e)}"

    def preview_text(file, use_ocr, ocr_language, quality, max_ocr_pages):
        """Preview extracted text"""
        nonlocal summarizer

        if summarizer is None:
            return "❌ 请先设置您的DeepSeek API密钥！Please set your DeepSeek API key first!"

        if file is None:
            return "❌ 请上传文件！Please upload a file!"

        try:
            # Temporarily disable OCR if requested
            original_ocr_state = summarizer.ocr_available
            if not use_ocr:
                summarizer.ocr_available = False

            text = summarizer.get_file_text(
                file.name,
                ocr_language=ocr_language,
                quality=quality,
                max_ocr_pages=max_ocr_pages
            )

            # Restore OCR state
            summarizer.ocr_available = original_ocr_state

            if text.startswith("Error") or text.startswith("❌"):
                return text

            # Show preview (first 2000 characters)
            preview = text[:2000] + "..." if len(text) > 2000 else text

            return f"""📄 **文本预览 Text Preview:**

总长度 Total Length: {len(text)} 字符 characters
预计块数 Estimated Chunks: {len(summarizer.text_splitter.split_text(text))}

--- 预览 Preview ---
{preview}
"""
        except Exception as e:
            return f"❌ 错误 Error: {str(e)}"

    def process_document(file, summary_type, include_quotes, use_ocr, ocr_language,
                         output_language, quality, max_ocr_pages, progress=gr.Progress()):
        """Process document with progress tracking"""
        nonlocal summarizer

        if summarizer is None:
            return "❌ 请先设置您的DeepSeek API密钥！Please set your DeepSeek API key first!"

        if file is None:
            return "❌ 请上传文件！Please upload a file!"

        start_time = time.time()

        try:
            # Progress callback
            def update_progress(value, desc):
                elapsed = time.time() - start_time
                progress(value, desc=f"{desc} (已用时 Elapsed: {elapsed:.1f}s)")

            # Extract text
            progress(0.1, desc="开始提取文本... Starting text extraction...")

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

            if text.startswith("Error") or text.startswith("❌") or text.startswith("用户已取消"):
                return text

            if len(text.strip()) < 10:
                return "❌ 文档中未找到可读文本。No readable text found in the document."

            # Show text statistics
            progress(0.5, desc=f"文本提取完成，长度: {len(text)} 字符 Text extracted, length: {len(text)} characters")

            # Generate summary
            progress(0.5, desc="生成摘要... Generating summary...")

            summary = summarizer.summarize_text_streaming(
                text,
                summary_type,
                include_quotes,
                output_language,
                progress_callback=update_progress
            )

            elapsed_time = time.time() - start_time
            progress(1.0, desc=f"完成！总用时: {elapsed_time:.1f}秒 Complete! Total time: {elapsed_time:.1f}s")

            # Add processing stats
            stats = f"\n\n---\n⏱️ 处理统计 Processing Stats:\n"
            stats += f"• 总用时 Total time: {elapsed_time:.1f} 秒 seconds\n"
            stats += f"• 文本长度 Text length: {len(text):,} 字符 characters\n"
            stats += f"• 文档块数 Document chunks: {len(summarizer.text_splitter.split_text(text))}\n"

            return summary + stats

        except Exception as e:
            elapsed_time = time.time() - start_time
            return f"❌ 错误 Error: {str(e)}\n⏱️ 失败时间 Failed after: {elapsed_time:.1f}秒 seconds"

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

                return "✅ 缓存清除成功！Cache cleared successfully!"
        except Exception as e:
            return f"❌ 清除缓存时出错 Error clearing cache: {str(e)}"

    def cancel_processing():
        """Cancel current processing"""
        nonlocal summarizer
        if summarizer:
            summarizer.cancel_current_processing()
            return "⚠️ 已请求取消处理... Processing cancellation requested..."
        return "没有活动的处理可取消 No active processing to cancel"

    # Create the interface
    with gr.Blocks(title="优化的文档摘要生成器 Optimized Document Summarizer", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # ⚡ 高速优化文档摘要生成器 (修复版)
            # ⚡ Optimized Document Summarizer with Timeout Protection
            ## 支持中英文的智能文档分析工具 | Intelligent Document Analysis Tool with Bilingual Support

            **🔧 主要修复 Main Fixes:**
            - ⏱️ API调用超时保护（60秒）| API call timeout protection (60s)
            - 📏 文本长度限制（100k字符）| Text length limit (100k characters)
            - 🔢 文档块数限制（最多20块）| Document chunk limit (max 20)
            - 💾 更好的错误处理和恢复 | Better error handling and recovery
            - 📊 实时处理统计 | Real-time processing statistics
            - 🚀 优化的处理流程 | Optimized processing flow
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔑 API配置 API Configuration")
                api_key_input = gr.Textbox(
                    label="DeepSeek API密钥 DeepSeek API Key",
                    placeholder="输入您的DeepSeek API密钥... Enter your DeepSeek API key...",
                    type="password"
                )
                api_key_button = gr.Button("设置API密钥 Set API Key", variant="primary")
                api_key_status = gr.Textbox(label="状态 Status", interactive=False)

                # Cache control
                gr.Markdown("### 💾 缓存控制 Cache Control")
                clear_cache_button = gr.Button("清除缓存 Clear Cache", variant="secondary")
                cache_status = gr.Textbox(label="缓存状态 Cache Status", interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("### 📤 文档上传 Document Upload")
                file_input = gr.File(
                    label="上传文档 Upload Document",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"],
                    type="filepath"
                )

                with gr.Row():
                    analyze_button = gr.Button("📊 快速分析 Quick Analysis", variant="secondary")
                    preview_button = gr.Button("👁️ 预览文本 Preview Text", variant="secondary")

                analysis_output = gr.Markdown()

        with gr.Row():
            with gr.Column():
                summary_type = gr.Radio(
                    choices=[
                        ("📝 简洁 Concise (推荐 Recommended)", "concise"),
                        ("📖 详细 Detailed", "detailed"),
                        ("• 要点 Bullet Points", "bullet_points"),
                        ("💡 关键见解 Key Insights", "key_insights"),
                        ("📑 章节式 Chapter-wise", "chapter_wise")
                    ],
                    value="concise",
                    label="摘要类型 Summary Type"
                )

                # Performance settings
                gr.Markdown("### ⚡ 性能设置 Performance Settings")

                quality = gr.Radio(
                    choices=[
                        ("🚀 快速 Fast (100 DPI)", "fast"),
                        ("⚖️ 平衡 Balanced (150 DPI)", "balanced"),
                        ("🎯 高质量 High Quality (200 DPI)", "high")
                    ],
                    value="fast",  # Changed default to fast
                    label="处理质量 Processing Quality"
                )

                max_ocr_pages = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=10,  # Reduced default
                    step=1,
                    label="最大OCR页数 Maximum OCR Pages",
                    info="仅在启用OCR时使用 Only used when OCR is enabled"
                )

                include_quotes = gr.Checkbox(
                    label="包含引用 Include quotes",
                    value=False  # Changed default to False
                )

                use_ocr = gr.Checkbox(
                    label="🔍 启用OCR Enable OCR (扫描文档 for scanned docs)",
                    value=False  # Default to False
                )

                ocr_language = gr.Radio(
                    choices=[
                        ("自动检测 Auto-detect", "auto"),
                        ("中文优先 Chinese Priority", "chinese"),
                        ("仅英文 English Only", "english")
                    ],
                    value="auto",
                    label="OCR语言 OCR Language",
                    visible=False  # Hide by default
                )

                output_language = gr.Radio(
                    choices=[
                        ("自动 Auto", "auto"),
                        ("中文 Chinese", "chinese"),
                        ("英文 English", "english")
                    ],
                    value="auto",
                    label="输出语言 Output Language"
                )

                with gr.Row():
                    summarize_button = gr.Button("🚀 生成摘要 Generate Summary", variant="primary", size="lg")
                    cancel_button = gr.Button("⏹️ 取消 Cancel", variant="stop", size="sm")

        gr.Markdown("### 📋 摘要输出 Summary Output")
        output_text = gr.Textbox(
            label="摘要 Summary",
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

        preview_button.click(
            fn=preview_text,
            inputs=[file_input, use_ocr, ocr_language, quality, max_ocr_pages],
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

        # Show/hide OCR language when OCR is toggled
        use_ocr.change(
            fn=lambda x: gr.update(visible=x),
            inputs=[use_ocr],
            outputs=[ocr_language]
        )

        gr.Markdown(
            """
            ### 🚀 快速开始 Quick Start:

            1. **设置API密钥** Set your DeepSeek API key
            2. **上传文档** Upload your document
            3. **点击"快速分析"查看文档信息** Click "Quick Analysis" to check document info
            4. **选择"简洁"摘要类型** Select "Concise" summary type
            5. **点击"生成摘要"** Click "Generate Summary"

            ### ⚠️ 如果处理时间过长 If Processing Takes Too Long:

            - **禁用OCR** Disable OCR if your PDF has selectable text
            - **使用"简洁"模式** Use "Concise" mode
            - **检查文档大小** Check document size in analysis
            - **考虑分割大文档** Consider splitting large documents
            - **点击"取消"停止处理** Click "Cancel" to stop processing

            ### 📊 性能基准 Performance Benchmarks:

            - 10页PDF（无OCR）: ~10-30秒 10-page PDF (no OCR): ~10-30s
            - 50页PDF（无OCR）: ~30-60秒 50-page PDF (no OCR): ~30-60s
            - 100页PDF（无OCR）: ~60-120秒 100-page PDF (no OCR): ~60-120s
            - OCR处理: 每页+20-30秒 OCR processing: +20-30s per page

            ### 🔧 技术限制 Technical Limits:

            - 最大文本: 100,000字符 Max text: 100,000 characters
            - 最大块数: 20 Max chunks: 20
            - API超时: 60秒 API timeout: 60s
            - 总超时: 300秒 Total timeout: 300s
            """
        )

    return interface


if __name__ == "__main__":
    print("""
    ====================================
    优化的文档摘要生成器 (修复版)
    OPTIMIZED DOCUMENT SUMMARIZER (FIXED)
    ====================================

    主要修复 Main fixes:
    - API调用超时保护 API call timeout protection
    - 文本长度限制 Text length limits
    - 文档块数限制 Document chunk limits
    - 更好的错误处理 Better error handling
    - 优化的默认设置 Optimized default settings
    - 文本预览功能 Text preview feature
    - 处理时间统计 Processing time statistics

    ====================================
    """)

    interface = create_optimized_gradio_interface()
    interface.launch(
        share=True,
        server_name="localhost",
        server_port=7860,
        show_error=True
    )