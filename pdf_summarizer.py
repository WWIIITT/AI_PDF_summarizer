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
    print("⚠️ 未安装OCR依赖项。OCR功能将被禁用。")

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    # Try to find punkt tokenizer
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

# punkt_tab is included with punkt in newer versions
# No need to download separately


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
                progress_callback(1.0, "从缓存加载 Loaded from cache")
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
                        return "用户已取消处理。Processing cancelled by user."

                    page_text = pdf.pages[i].extract_text() or ""
                    if self.is_scanned_pdf_page(page_text) or self.is_text_corrupted(page_text):
                        needs_ocr = True
                        break

                if not needs_ocr:
                    # Extract all text quickly
                    for i, page in enumerate(pdf.pages):
                        if self.cancel_processing:
                            return "用户已取消处理。Processing cancelled by user."

                        if progress_callback:
                            progress_callback(i / total_pages, f"提取第 {i + 1}/{total_pages} 页 Extracting page {i + 1}/{total_pages}")

                        page_text = page.extract_text() or ""
                        if page_text and not self.is_text_corrupted(page_text):
                            extracted_text += f"\n--- 第 {i + 1}/{total_pages} 页 Page {i + 1}/{total_pages} ---\n{page_text}\n"

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
                progress_callback(0.1, f"PDF共有 {total_pages} 页，将OCR处理前 {max_ocr_pages} 页... PDF has {total_pages} pages. Will OCR up to {max_ocr_pages} pages...")

            # Limit pages to process
            pages_to_process = min(total_pages, max_ocr_pages)

            # Convert only the pages we need
            extracted_text = ""

            # Process pages in smaller batches to avoid memory issues
            batch_size = 5

            for batch_start in range(0, pages_to_process, batch_size):
                if self.cancel_processing:
                    return "用户已取消处理。Processing cancelled by user."

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
                        fmt='jpeg',  # JPEG is faster than PNG
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
                    print(f"转换批次 {batch_start}-{batch_end} 时出错 Error converting batch {batch_start}-{batch_end}: {str(e)}")
                    continue

            # Add note about remaining pages if any
            if total_pages > pages_to_process:
                extracted_text += f"\n\n--- 注意：OCR仅处理了前 {pages_to_process} 页，共 {total_pages} 页 Note: OCR processed first {pages_to_process} pages out of {total_pages} total pages ---\n"

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

                for i, page in enumerate(pdf_reader.pages):
                    if self.cancel_processing:
                        return "用户已取消处理。Processing cancelled by user."

                    if progress_callback:
                        progress_callback(i / total_pages, f"提取第 {i + 1}/{total_pages} 页 Extracting page {i + 1}/{total_pages}")

                    try:
                        page_text = page.extract_text()
                        if page_text and not self.is_text_corrupted(page_text):
                            extracted_text += f"\n--- 第 {i + 1}/{total_pages} 页 Page {i + 1}/{total_pages} ---\n{page_text}\n"
                    except:
                        continue

            return extracted_text if extracted_text else "无法从PDF中提取文本。No text could be extracted from the PDF."
        except Exception as e:
            return f"读取PDF时出错 Error reading PDF: {str(e)}"

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
            return "OCR不可用。OCR not available."

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
                            return file.read()
                    except UnicodeDecodeError:
                        continue
                return "错误：无法解码文本文件。Error: Unable to decode text file."
            except Exception as e:
                return f"读取文本文件时出错 Error reading text file: {str(e)}"
        else:
            return "不支持的文件格式。Unsupported file format."

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
            return "未找到可总结的内容。No content found to summarize."

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
            "chinese": "用中文撰写摘要。Write the summary in Chinese (中文).",
            "english": "用英文撰写摘要。Write the summary in English.",
            "auto": "使用源文档的语言撰写摘要。Match the language of the source document."
        }
        lang_instruction = lang_instructions.get(output_language, lang_instructions["auto"])

        # Summary prompts
        prompts = {
            "concise": f"撰写一个简洁的2-3段摘要。Write a concise 2-3 paragraph summary. {lang_instruction}\n\n{{text}}\n\n摘要 SUMMARY:",
            "detailed": f"创建一个包含关键引用的综合摘要。Create a comprehensive summary with key quotes if available. {lang_instruction}\n\n{{text}}\n\n详细摘要 DETAILED SUMMARY:",
            "bullet_points": f"创建要点式摘要。Create a bullet-point summary. {lang_instruction}\n\n{{text}}\n\n要点 BULLET POINTS:",
            "key_insights": f"提取5-7个关键见解。Extract 5-7 key insights. {lang_instruction}\n\n{{text}}\n\n关键见解 KEY INSIGHTS:",
            "chapter_wise": f"创建逐章节摘要。Create a section-by-section summary. {lang_instruction}\n\n{{text}}\n\n章节摘要 SECTION SUMMARY:"
        }

        prompt_template = prompts.get(summary_type, prompts["concise"])

        try:
            if len(documents) > 1:
                # Use map-reduce for long documents
                if progress_callback:
                    progress_callback(0.3, "处理文档块... Processing document chunks...")

                map_prompt = PromptTemplate(
                    template=f"总结这个部分。Summarize this section. {lang_instruction}\n\n{{text}}\n\n摘要 SUMMARY:",
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
                    SystemMessage(content="你是精通多种语言的专业文档分析师。You are an expert document analyst fluent in multiple languages."),
                    HumanMessage(content=prompt_template.format(text=documents[0].page_content))
                ]

                # Stream the response
                summary_parts = []
                for chunk in self.llm.stream(messages):
                    summary_parts.append(chunk.content)
                    if progress_callback:
                        progress_callback(0.5 + 0.5 * (len(summary_parts) / 100), "生成摘要... Generating summary...")

                summary = "".join(summary_parts)

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
            "total_sentences": text.count('.') + text.count('。'),
            "detected_language": "中文 Chinese" if len(re.findall(r'[\u4e00-\u9fff]', text[:1000])) > 100 else "英文 English",
            "text_quality": "损坏 Corrupted" if self.is_text_corrupted(text) else "良好 Good",
            "recommended_summary": "详细 detailed" if len(text.split()) > 5000 else "简洁 concise"
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
• **检测语言 Detected Language:** {analysis['detected_language']}
• **文本质量 Text Quality:** {analysis['text_quality']}
• **推荐摘要类型 Recommended Summary:** {analysis['recommended_summary']}

💡 **性能提示 Performance Tips:**
• 对于大型PDF（>50页），考虑限制OCR页数 For large PDFs (>50 pages), consider limiting OCR pages
• 使用"快速"质量获得快速结果 Use 'Fast' quality for quick results
• 使用"平衡"获得最佳速度/质量平衡 Use 'Balanced' for optimal speed/quality
• 启用缓存以重复处理 Enable caching for repeated processing

⚠️ **注意 Note:** 如果文档是扫描件，OCR处理可能需要几分钟。If the document is scanned, OCR processing may take several minutes.
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

        try:
            # Progress callback
            def update_progress(value, desc):
                progress(value, desc=desc)

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

            if text.startswith("Error") or text.startswith("❌"):
                return text

            if len(text.strip()) < 10:
                return "❌ 文档中未找到可读文本。No readable text found in the document."

            # Generate summary
            progress(0.5, desc="生成摘要... Generating summary...")

            summary = summarizer.summarize_text_streaming(
                text,
                summary_type,
                include_quotes,
                output_language,
                progress_callback=update_progress
            )

            progress(1.0, desc="完成！Complete!")

            return summary

        except Exception as e:
            return f"❌ 错误 Error: {str(e)}"

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
            # ⚡ 高速优化文档摘要生成器
            # ⚡ Optimized Document Summarizer with High-Speed Processing
            ## 支持中英文OCR的智能文档分析工具 | Intelligent Document Analysis Tool with Chinese & English OCR Support

            **🚀 性能特点 Performance Features:**
            - ⚡ 带超时保护的并行OCR处理 | Parallel OCR processing with timeout protection
            - 💾 智能缓存重复文档 | Intelligent caching for repeated documents
            - 🔄 流式响应更快反馈 | Streaming responses for faster feedback
            - 🎯 速度/准确度平衡的质量设置 | Quality settings for speed/accuracy balance
            - 📊 实时进度跟踪 | Real-time progress tracking
            - ⏹️ 可取消操作 | Cancellable operations
            - 🔢 OCR页面限制控制 | Page limit controls for OCR
            - 🇨🇳 中英文双语支持 | Bilingual Chinese-English support
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

                analyze_button = gr.Button("📊 快速分析 Quick Analysis", variant="secondary")
                analysis_output = gr.Markdown()

        with gr.Row():
            with gr.Column():
                summary_type = gr.Radio(
                    choices=[
                        ("📝 简洁 Concise", "concise"),
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
                    value="balanced",
                    label="处理质量 Processing Quality"
                )

                max_ocr_pages = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=20,
                    step=1,
                    label="最大OCR页数（用于大型PDF）Maximum OCR Pages (for large PDFs)",
                    info="限制OCR处理前N页以避免超时 Limit OCR processing to first N pages to avoid timeout"
                )

                include_quotes = gr.Checkbox(
                    label="包含引用（详细模式）Include quotes (detailed mode)",
                    value=True
                )

                use_ocr = gr.Checkbox(
                    label="🔍 启用OCR Enable OCR",
                    value=False
                )

                ocr_language = gr.Radio(
                    choices=[
                        ("自动检测 Auto-detect", "auto"),
                        ("中文优先 Chinese Priority", "chinese"),
                        ("仅英文 English Only", "english")
                    ],
                    value="auto",
                    label="OCR语言 OCR Language"
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
            ### ⚡ 性能优化提示 Performance Tips:

            1. **对于带OCR的大型PDF For large PDFs with OCR:**
               - 限制OCR页数（例如10-20页）以避免超时 Limit OCR pages (e.g., 10-20 pages) to avoid timeout
               - 使用快速模式进行初始测试 Use Fast mode for initial testing
               - 考虑分批处理 Consider processing in batches

            2. **如果OCR耗时过长 If OCR is taking too long:**
               - 点击取消按钮停止处理 Click Cancel button to stop processing
               - 尝试更少的页数 Try with fewer pages
               - 使用快速质量设置 Use Fast quality setting
               - 如果文本已可选择，禁用OCR Disable OCR if text is already selectable

            3. **一般提示 General tips:**
               - 启用缓存 - 重新处理缓存文档是即时的 Enable caching - Reprocessing cached documents is instant
               - 使用平衡模式获得最佳速度/质量权衡 Use Balanced mode for optimal speed/quality trade-off
               - 仅对扫描质量差的关键文档使用高质量模式 High Quality mode only for critical documents with poor scan quality

            ### 🛠️ 技术优化 Technical Optimizations:
            - OCR超时保护（每页30秒）OCR timeout protection (30 seconds per page)
            - 内存高效的批处理 Memory-efficient batch processing
            - 页面限制控制 Page limit controls
            - 降低DPI设置以加快处理速度 Reduced DPI settings for faster processing
            - 可取消操作 Cancellable operations
            - 智能缓存系统 Smart caching system
            - 优化的图像预处理 Optimized image preprocessing

            ### 📞 支持信息 Support Information:
            - 确保已安装Tesseract OCR Ensure Tesseract OCR is installed
            - 中文OCR需要chi_sim语言包 Chinese OCR requires chi_sim language pack
            - 大文件建议使用分批处理 Large files recommended to process in batches
            """
        )

    return interface


if __name__ == "__main__":
    print("""
    ====================================
    优化的文档摘要生成器
    OPTIMIZED DOCUMENT SUMMARIZER
    ====================================

    主要修复OCR超时问题的功能：
    Key fixes for OCR timeout issues:
    - 每页30秒超时保护 Added 30-second timeout per page
    - 页面限制控制（默认：20页）Page limit control (default: 20 pages)
    - 降低DPI设置（100-200）Reduced DPI settings (100-200)
    - 批处理以避免内存问题 Batch processing to avoid memory issues
    - 可取消操作 Cancellable operations
    - 更好的错误处理 Better error handling

    安装要求 Installation:
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