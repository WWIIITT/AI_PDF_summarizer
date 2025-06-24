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
from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import sent_tokenize
import os
import tempfile
import pdfplumber  # Better for Chinese PDFs

# Try to import OCR dependencies
OCR_AVAILABLE = False
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    import numpy as np
    import cv2

    OCR_AVAILABLE = True
except ImportError:
    print("⚠️ OCR dependencies not installed. OCR features will be disabled.")
    print("To enable OCR, install: pip install pytesseract pdf2image pillow opencv-python-headless")

warnings.filterwarnings('ignore')

# Download required NLTK data
required_data = ['punkt', 'punkt_tab']
for data in required_data:
    try:
        nltk.data.find(f'tokenizers/{data}')
    except LookupError:
        print(f"Downloading {data}...")
        nltk.download(data)


class EnhancedDocumentSummarizer:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(
            model='deepseek-chat',
            openai_api_key=api_key,
            openai_api_base='https://api.deepseek.com',
            max_tokens=2048,
            temperature=0.3
        )

        # Enhanced text splitter that respects sentence boundaries
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=400,
            length_function=len,
            separators=["\n\n", "\n", "。", ". ", "！", "! ", "？", "? ", "；", "; ", " ", ""],
            is_separator_regex=False
        )

        # Splitter for extracting quotable segments
        self.quote_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )

        # OCR configuration
        self.ocr_available = False
        self.chinese_ocr_available = False
        if OCR_AVAILABLE:
            self.configure_ocr()

    def configure_ocr(self):
        """Configure OCR settings and check if Tesseract is installed"""
        try:
            # Test if tesseract is installed
            pytesseract.get_tesseract_version()
            self.ocr_available = True

            # Check for available languages
            available_langs = pytesseract.get_languages()
            self.chinese_ocr_available = any(lang in available_langs for lang in ['chi_sim', 'chi_tra'])

            if not self.chinese_ocr_available:
                print("⚠️ Chinese language packs not found for Tesseract.")
                print("  To scan Chinese documents, install language packs:")
                print("  - Windows: Download chi_sim.traineddata and chi_tra.traineddata")
                print("  - Mac: brew install tesseract-lang")
                print("  - Linux: sudo apt-get install tesseract-ocr-chi-sim tesseract-ocr-chi-tra")
        except:
            self.ocr_available = False
            self.chinese_ocr_available = False
            print("⚠️ Tesseract OCR not found. Please install it:")
            print("  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
            print("  - Mac: brew install tesseract")
            print("  - Linux: sudo apt-get install tesseract-ocr")

    def preprocess_image_for_ocr(self, image):
        """Preprocess image to improve OCR accuracy"""
        if not OCR_AVAILABLE:
            return image

        # Convert PIL Image to numpy array
        img_array = np.array(image)

        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array

        # Apply thresholding to get better OCR results
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Denoise
        denoised = cv2.medianBlur(thresh, 3)

        # Convert back to PIL Image
        return Image.fromarray(denoised)

    def extract_text_with_ocr(self, image, preprocess=True, language='auto'):
        """Extract text from image using OCR with Chinese support"""
        if not self.ocr_available or not OCR_AVAILABLE:
            return "OCR not available. Please install Tesseract and pytesseract."

        try:
            if preprocess:
                image = self.preprocess_image_for_ocr(image)

            # Determine OCR language settings
            if language == 'auto':
                # Use all available languages for automatic detection
                if self.chinese_ocr_available:
                    ocr_lang = 'eng+chi_sim+chi_tra'
                else:
                    ocr_lang = 'eng'
            elif language == 'chinese':
                if self.chinese_ocr_available:
                    ocr_lang = 'chi_sim+chi_tra+eng'
                else:
                    return "Chinese OCR not available. Please install Chinese language packs for Tesseract."
            elif language == 'english':
                ocr_lang = 'eng'
            else:
                ocr_lang = language

            # Perform OCR with multiple language support
            text = pytesseract.image_to_string(
                image,
                lang=ocr_lang,
                config='--psm 3 -c preserve_interword_spaces=1'  # Better handling of spacing
            )

            return text.strip()
        except Exception as e:
            return f"OCR Error: {str(e)}"

    def is_scanned_pdf_page(self, page_text):
        """Check if a PDF page is scanned (image-based) or has extractable text"""
        # If extracted text is very short or mostly whitespace, it's likely scanned
        return len(page_text.strip()) < 50

    def detect_language(self, text_sample):
        """Simple language detection based on character patterns"""
        if not text_sample:
            return 'auto'

        # Count Chinese characters
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text_sample))
        total_chars = len(text_sample)

        if chinese_chars > total_chars * 0.3:  # If more than 30% Chinese characters
            return 'chinese'
        else:
            return 'english'

    def extract_text_from_pdf_with_pdfplumber(self, file_path):
        """Try to extract text using pdfplumber (better for Chinese PDFs)"""
        try:
            import pdfplumber
            extracted_text = ""

            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)

                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        # Clean up extracted text
                        page_text = re.sub(r'\n+', '\n', page_text)
                        page_text = re.sub(r' +', ' ', page_text)
                        extracted_text += f"\n--- Page {i + 1}/{total_pages} ---\n{page_text}\n"

            return extracted_text if extracted_text else None
        except:
            return None

    def extract_text_from_pdf(self, file_path, use_ocr_if_needed=True, ocr_language='auto'):
        """Enhanced PDF text extraction with OCR support for scanned documents"""
        try:
            extracted_text = ""
            ocr_pages = []

            # First try pdfplumber for better Chinese support
            pdfplumber_text = None
            try:
                import pdfplumber
                pdfplumber_text = self.extract_text_from_pdf_with_pdfplumber(file_path)
                if pdfplumber_text and len(pdfplumber_text.strip()) > 100:
                    # Check if the extracted text is readable
                    if not self.is_text_corrupted(pdfplumber_text):
                        return pdfplumber_text
            except ImportError:
                print(
                    "Note: pdfplumber not installed. Install it for better Chinese PDF support: pip install pdfplumber")

            # If pdfplumber failed or text is corrupted, try PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)

                # Sample text for language detection
                sample_text = ""

                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                    except:
                        page_text = ""

                    if i < 3:  # Sample first 3 pages for language detection
                        sample_text += page_text

                    # Check if page needs OCR or if text is corrupted
                    if use_ocr_if_needed and (self.is_scanned_pdf_page(page_text) or self.is_text_corrupted(page_text)):
                        ocr_pages.append(i + 1)
                    else:
                        # Clean up regular extracted text
                        page_text = re.sub(r'\n+', '\n', page_text)
                        page_text = re.sub(r' +', ' ', page_text)

                        # Check again if text is readable
                        if not self.is_text_corrupted(page_text):
                            extracted_text += f"\n--- Page {i + 1}/{total_pages} ---\n{page_text}\n"

            # Auto-detect language if needed
            if ocr_language == 'auto' and sample_text:
                ocr_language = self.detect_language(sample_text)

            # If all pages need OCR or text is corrupted
            if (len(ocr_pages) == total_pages or self.is_text_corrupted(extracted_text)) and not self.ocr_available:
                return """❌ This appears to be a scanned PDF or contains non-standard encoding that requires OCR to read properly.

The PDF contains text that cannot be extracted without OCR tools.

To process this document, please:
1. Install OCR dependencies:
   pip install pytesseract pdf2image pillow opencv-python-headless pdfplumber

2. Install Tesseract OCR:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: brew install tesseract tesseract-lang
   - Linux: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra

3. For better Chinese PDF support without OCR:
   pip install pdfplumber

After installation, restart the application and try again."""

            # If OCR is needed and available
            if ocr_pages and use_ocr_if_needed and self.ocr_available and OCR_AVAILABLE:
                print(f"📸 Performing OCR on {len(ocr_pages)} pages...")
                print(f"🌐 Language mode: {ocr_language}")

                # Convert PDF to images for OCR
                images = convert_from_path(file_path, dpi=300)

                for page_num in ocr_pages:
                    if page_num <= len(images):
                        image = images[page_num - 1]
                        ocr_text = self.extract_text_with_ocr(image, language=ocr_language)
                        extracted_text += f"\n--- Page {page_num}/{total_pages} (OCR) ---\n{ocr_text}\n"

            # Final check
            if not extracted_text.strip() or self.is_text_corrupted(extracted_text):
                return """❌ Unable to extract readable text from this PDF.

This appears to be:
- A scanned document requiring OCR, or
- A PDF with special encoding/fonts that standard tools cannot read

Please ensure OCR tools are installed (see instructions above) or try converting the PDF to a different format."""

            return extracted_text.strip()

        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def is_text_corrupted(self, text):
        """Check if extracted text is corrupted or unreadable"""
        if not text or len(text.strip()) < 10:
            return True

        # Count readable characters vs special/control characters
        readable_chars = len(re.findall(r'[\u4e00-\u9fff\u0020-\u007E\u00A0-\u00FF]', text))
        total_chars = len(text)

        # If less than 30% of characters are readable, it's likely corrupted
        if total_chars > 0 and readable_chars / total_chars < 0.3:
            return True

        # Check for excessive special characters or patterns indicating corruption
        corruption_patterns = [
            r'[\x00-\x1F\x7F-\x9F]{5,}',  # Control characters
            r'[^\u4e00-\u9fff\u0020-\u007E\u00A0-\u00FF\s]{10,}',  # Long sequences of special chars
        ]

        for pattern in corruption_patterns:
            if re.search(pattern, text[:1000]):  # Check first 1000 chars
                return True

        return False

    def extract_text_from_image(self, file_path, ocr_language='auto'):
        """Extract text from image files using OCR"""
        if not OCR_AVAILABLE:
            return "❌ OCR dependencies not installed. Cannot process image files."

        try:
            image = Image.open(file_path)
            text = self.extract_text_with_ocr(image, language=ocr_language)
            return text if text else "No text could be extracted from the image."
        except Exception as e:
            return f"Error reading image: {str(e)}"

    def extract_text_from_docx(self, file_path):
        """Enhanced Word document extraction with formatting awareness"""
        try:
            doc = docx.Document(file_path)
            text = ""

            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    if paragraph.style.name.startswith('Heading'):
                        text += f"\n## {paragraph.text}\n"
                    else:
                        text += paragraph.text + "\n"

            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join([cell.text.strip() for cell in row.cells])
                    if row_text.strip():
                        text += row_text + "\n"

            return text.strip() if text else "No text found in the document."
        except Exception as e:
            return f"Error reading Word document: {str(e)}"

    def get_file_text(self, file_path, ocr_language='auto'):
        """Extract text based on file extension with OCR support"""
        file_lower = file_path.lower()

        # PDF files
        if file_lower.endswith('.pdf'):
            return self.extract_text_from_pdf(file_path, ocr_language=ocr_language)

        # Word documents
        elif file_lower.endswith(('.docx', '.doc')):
            return self.extract_text_from_docx(file_path)

        # Image files
        elif file_lower.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            return self.extract_text_from_image(file_path, ocr_language=ocr_language)

        # Text files
        elif file_lower.endswith('.txt'):
            try:
                # Try UTF-8 first, then fallback to other encodings
                encodings = ['utf-8', 'gbk', 'gb2312', 'big5', 'utf-16']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as file:
                            return file.read()
                    except UnicodeDecodeError:
                        continue
                return "Error: Unable to decode text file with supported encodings."
            except Exception as e:
                return f"Error reading text file: {str(e)}"

        else:
            return "Unsupported file format. Supported formats: PDF, Word (.docx, .doc), Images (PNG, JPG, JPEG, TIFF, BMP, GIF), and Text files."

    def extract_key_sentences(self, text: str, num_sentences: int = 5) -> List[str]:
        """Extract potentially important sentences for quoting"""
        try:
            # For Chinese text, also split by Chinese punctuation
            if re.search(r'[\u4e00-\u9fff]', text):
                # Chinese sentence splitting
                sentences = re.split(r'[。！？]', text)
                sentences = [s.strip() for s in sentences if s.strip()]
            else:
                sentences = sent_tokenize(text)

            valid_sentences = [s for s in sentences if len(s) > 10]
            sorted_sentences = sorted(valid_sentences, key=len, reverse=True)
            step = max(1, len(sorted_sentences) // num_sentences)
            key_sentences = sorted_sentences[::step][:num_sentences]
            return key_sentences
        except:
            sentences = re.split(r'[.。！!？?]', text)
            return [s.strip() for s in sentences if s.strip()][:num_sentences]

    def create_detailed_summary_prompt(self, include_quotes: bool = True, output_language: str = "auto"):
        """Create enhanced prompt for detailed summaries with language control"""
        # Language instructions based on user preference
        if output_language == "chinese":
            lang_instruction = "Provide the summary in Chinese (中文), regardless of the source language."
        elif output_language == "english":
            lang_instruction = "Provide the summary in English, regardless of the source language."
        else:
            lang_instruction = "If the text is in Chinese, provide the summary in Chinese; if in English, summarize in English."

        if include_quotes:
            return f"""You are an expert document analyst fluent in both English and Chinese. Create a comprehensive summary of the following text.

INSTRUCTIONS:
1. Provide a detailed summary covering all major points and important details
2. Include 3-5 relevant direct quotes from the text that support key points
3. Format quotes as: "quote text" (from the document)
4. Organize the summary with clear sections if the content has multiple topics
5. Highlight any critical findings, conclusions, or recommendations
6. Preserve important numbers, dates, and specific facts
7. {lang_instruction}

TEXT TO SUMMARIZE:
{{text}}

DETAILED SUMMARY WITH QUOTES:"""
        else:
            return f"""Create a detailed summary of the following text, including all key points and important details. 
{lang_instruction}

{{text}}

DETAILED SUMMARY:"""

    def summarize_text(self, text, summary_type="concise", include_quotes=False, output_language="auto"):
        """Enhanced summarization with quote extraction for detailed mode and language control"""
        if not text or text.startswith("Error") or text.startswith("❌") or text.startswith("Unsupported"):
            return text

        # Create documents from text chunks
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        if not documents:
            return "No content found to summarize."

        # Language instructions based on user preference
        if output_language == "chinese":
            lang_instruction = "Write the summary in Chinese (中文), regardless of the source language."
        elif output_language == "english":
            lang_instruction = "Write the summary in English, regardless of the source language."
        else:
            lang_instruction = "If the text is primarily in Chinese, write the summary in Chinese. If primarily in English, write in English."

        # Define different summary prompts
        prompts = {
            "concise": f"""Write a concise summary of the following text in 2-3 paragraphs.
{lang_instruction}

{{text}}

CONCISE SUMMARY:""",

            "detailed": self.create_detailed_summary_prompt(include_quotes, output_language),

            "bullet_points": f"""Create a comprehensive bullet-point summary of the following text:

INSTRUCTIONS:
- Use main bullets for major topics
- Use sub-bullets for supporting details
- Include important facts, figures, and dates
- Organize logically by theme or chronology
- {lang_instruction}

{{text}}

BULLET POINT SUMMARY:""",

            "key_insights": f"""Extract the key insights and takeaways from the following text:

INSTRUCTIONS:
1. Identify 5-7 most important insights
2. Explain why each insight matters
3. Include any actionable recommendations
4. Note any surprising findings
5. {lang_instruction}

{{text}}

KEY INSIGHTS:""",

            "chapter_wise": f"""Create a chapter-by-chapter or section-by-section summary:

INSTRUCTIONS:
1. Identify major sections or chapters
2. Summarize each section separately
3. Note key themes that connect sections
4. Include transitions between topics
5. {lang_instruction}

{{text}}

CHAPTER-WISE SUMMARY:"""
        }

        prompt_template = prompts.get(summary_type, prompts["concise"])

        try:
            # For longer documents, use map-reduce with custom prompts
            if len(documents) > 1:
                map_prompt = PromptTemplate(
                    template=f"""Summarize this section of the document. {lang_instruction}

{{text}}

SECTION SUMMARY:""",
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
                messages = [
                    SystemMessage(
                        content="You are an expert document analyst fluent in multiple languages including English and Chinese. Create clear, informative, and well-structured summaries. Always follow the language instructions provided."),
                    HumanMessage(content=prompt_template.format(text=text))
                ]
                response = self.llm(messages)
                summary = response.content

            summary = self._format_summary(summary)
            return summary
        except Exception as e:
            return f"Error during summarization: {str(e)}"

    def _format_summary(self, summary: str) -> str:
        """Clean up and format the summary for better readability"""
        summary = re.sub(r'\n{3,}', '\n\n', summary)
        summary = re.sub(r'"\s*([^"]+)\s*"', r'"\1"', summary)
        return summary.strip()

    def analyze_document_structure(self, text: str) -> Dict[str, any]:
        """Analyze document structure for better summarization"""
        analysis = {
            "total_words": len(text.split()),
            "total_sentences": len(sent_tokenize(text)) if text else 0,
            "sections": [],
            "has_chapters": False,
            "has_headings": False,
            "ocr_quality": "N/A",
            "detected_language": "Unknown",
            "text_quality": "Good"
        }

        # Check if text is corrupted
        if self.is_text_corrupted(text):
            analysis["text_quality"] = "Corrupted - OCR Required"
            analysis["detected_language"] = "Unable to detect - text corrupted"
            return analysis

        # Detect language
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text[:1000]))  # Check first 1000 chars
        if chinese_chars > 100:
            analysis["detected_language"] = "Chinese"
        elif chinese_chars > 10:
            analysis["detected_language"] = "Mixed (Chinese & English)"
        else:
            analysis["detected_language"] = "English"

        # Check for OCR indicators
        if "(OCR)" in text:
            # Simple OCR quality check based on common OCR errors
            ocr_error_patterns = [r'\b[0-9]+[a-zA-Z]+[0-9]+\b', r'[|!l1]{3,}', r'[0oO]{3,}']
            error_count = sum(len(re.findall(pattern, text)) for pattern in ocr_error_patterns)

            if error_count < 10:
                analysis["ocr_quality"] = "Good"
            elif error_count < 50:
                analysis["ocr_quality"] = "Fair"
            else:
                analysis["ocr_quality"] = "Poor - may need manual review"

        # Check for chapter markers (including Chinese)
        chapter_patterns = [
            r'Chapter \d+',
            r'CHAPTER \d+',
            r'Section \d+',
            r'Part \d+',
            r'第[一二三四五六七八九十\d]+章',
            r'第[一二三四五六七八九十\d]+节',
            r'第[一二三四五六七八九十\d]+部分'
        ]

        for pattern in chapter_patterns:
            if re.search(pattern, text):
                analysis["has_chapters"] = True
                break

        # Check for heading patterns
        if re.search(r'\n#{1,3} .+\n', text) or re.search(r'\n[A-Z][A-Z\s]+\n', text):
            analysis["has_headings"] = True

        return analysis


def create_enhanced_gradio_interface():
    """Create the enhanced Gradio interface with OCR support and language selection"""

    summarizer = None

    def set_api_key(api_key):
        """Set the API key and initialize the summarizer"""
        global summarizer
        if api_key.strip():
            try:
                summarizer = EnhancedDocumentSummarizer(api_key.strip())
                ocr_status = "✅ OCR Available" if summarizer.ocr_available else "⚠️ OCR Not Available"
                chinese_status = "✅ Chinese OCR Ready" if summarizer.chinese_ocr_available else "⚠️ Chinese OCR Not Ready"

                if not summarizer.ocr_available:
                    status_msg = f"""✅ API Key set successfully! | {ocr_status} | {chinese_status}

⚠️ **Note**: OCR is not available. To process scanned PDFs or images:
1. Install Python packages: `pip install pytesseract pdf2image pillow opencv-python-headless`
2. Install Tesseract OCR software:
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   - Mac: brew install tesseract tesseract-lang
   - Linux: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra
3. For better Chinese PDF support: `pip install pdfplumber`"""
                else:
                    status_msg = f"✅ API Key set successfully! | {ocr_status} | {chinese_status}"

                return status_msg
            except Exception as e:
                return f"❌ Error setting API key: {str(e)}"
        else:
            return "❌ Please enter a valid API key"

    def analyze_document(file):
        """Analyze document before summarization"""
        global summarizer

        if summarizer is None:
            return "❌ Please set your DeepSeek API key first!"

        if file is None:
            return "❌ Please upload a file!"

        try:
            text = summarizer.get_file_text(file.name)
            if text.startswith("Error") or text.startswith("Unsupported") or text.startswith("❌"):
                return text

            analysis = summarizer.analyze_document_structure(text)

            result = f"""📊 **Document Analysis:**

• **Total Words:** {analysis['total_words']:,}
• **Total Sentences:** {analysis['total_sentences']:,}
• **Detected Language:** {analysis['detected_language']}
• **Text Quality:** {analysis['text_quality']}
• **Has Chapters/Sections:** {'Yes' if analysis['has_chapters'] else 'No'}
• **Has Clear Headings:** {'Yes' if analysis['has_headings'] else 'No'}
• **OCR Quality:** {analysis['ocr_quality']}
• **Recommended Summary Type:** {'chapter_wise' if analysis['has_chapters'] else 'detailed' if analysis['total_words'] > 5000 else 'concise'}

📝 **File Type:** {os.path.splitext(file.name)[1].upper()}
🔍 **OCR Status:** {'Used' if '(OCR)' in text else 'Not needed' if analysis['text_quality'] == 'Good' else 'Required but not available'}
🌐 **Chinese OCR Available:** {'Yes' if summarizer.chinese_ocr_available else 'No - Install language packs'}

"""
            if analysis['text_quality'] == 'Corrupted - OCR Required' and not summarizer.ocr_available:
                result += """
⚠️ **Warning**: This document requires OCR to be read properly. Please install OCR dependencies."""

            return result
        except Exception as e:
            return f"❌ Error analyzing document: {str(e)}"

    def process_document(file, summary_type, include_quotes, use_ocr, ocr_language, output_language,
                         progress=gr.Progress()):
        """Process the uploaded document and return summary"""
        global summarizer

        if summarizer is None:
            return "❌ Please set your DeepSeek API key first!"

        if file is None:
            return "❌ Please upload a file!"

        try:
            progress(0.2, desc="Extracting text from document...")

            # Temporarily disable OCR if requested
            original_ocr_state = summarizer.ocr_available
            if not use_ocr:
                summarizer.ocr_available = False

            # Extract text from the uploaded file
            text = summarizer.get_file_text(file.name, ocr_language=ocr_language)

            # Restore OCR state
            summarizer.ocr_available = original_ocr_state

            if text.startswith("Error") or text.startswith("Unsupported") or text.startswith("❌"):
                return text

            if len(text.strip()) < 10:
                return "❌ No readable text found in the document. If this is a scanned document, ensure OCR is enabled and Tesseract is installed."

            progress(0.6, desc="Generating summary...")

            # Generate summary with language preference
            summary = summarizer.summarize_text(text, summary_type, include_quotes, output_language)

            progress(1.0, desc="Complete!")

            return summary

        except Exception as e:
            return f"❌ Error processing document: {str(e)}"

    # Create the Gradio interface
    with gr.Blocks(title="Enhanced Document Summarizer with Chinese OCR", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 📚 Enhanced Document Summarizer with Chinese OCR Support
            ## 增强版文档摘要生成器（支持中文OCR）

            Upload documents (including scanned PDFs and images) and get AI-powered summaries in your preferred language.
            上传文档（包括扫描的PDF和图片）并获得您偏好语言的AI生成摘要。

            **✨ Features 功能特点:**
            - 🔍 OCR support for scanned documents and images (支持扫描文档和图片的OCR)
            - 🇨🇳 Chinese and English text recognition (中英文文本识别)
            - 🌐 Choose output language independently (独立选择输出语言)
            - 📄 Multiple file format support (多种文件格式支持)
            - 💬 Quote extraction in detailed mode (详细模式下的引用提取)
            - 📊 Document structure analysis (文档结构分析)
            - 🎯 Multiple summary formats (多种摘要格式)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🔑 API Configuration 配置")
                api_key_input = gr.Textbox(
                    label="DeepSeek API Key",
                    placeholder="Enter your DeepSeek API key...",
                    type="password"
                )
                api_key_button = gr.Button("Set API Key 设置密钥", variant="primary")
                api_key_status = gr.Textbox(label="Status 状态", interactive=False, lines=5)

            with gr.Column(scale=2):
                gr.Markdown("### 📤 Document Upload 文档上传")
                file_input = gr.File(
                    label="Upload Document 上传文档",
                    file_types=[".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"],
                    type="filepath"
                )

                analyze_button = gr.Button("📊 Analyze Document 分析文档", variant="secondary")
                analysis_output = gr.Markdown()

        with gr.Row():
            with gr.Column():
                summary_type = gr.Radio(
                    choices=[
                        ("📝 Concise - Brief overview 简洁概述", "concise"),
                        ("📖 Detailed - Comprehensive with quotes 详细摘要（含引用）", "detailed"),
                        ("• Bullet Points - Organized list 要点列表", "bullet_points"),
                        ("💡 Key Insights - Main takeaways 关键见解", "key_insights"),
                        ("📑 Chapter-wise - Section by section 章节摘要", "chapter_wise")
                    ],
                    value="concise",
                    label="Summary Type 摘要类型"
                )

                include_quotes = gr.Checkbox(
                    label="Include direct quotes 包含直接引用 (for detailed summary)",
                    value=True
                )

                use_ocr = gr.Checkbox(
                    label="🔍 Enable OCR for scanned documents 启用OCR扫描文档",
                    value=True
                )

                ocr_language = gr.Radio(
                    choices=[
                        ("Auto-detect 自动检测", "auto"),
                        ("Chinese Priority 中文优先", "chinese"),
                        ("English Only 仅英文", "english")
                    ],
                    value="auto",
                    label="OCR Language 语言设置"
                )

                # New output language selection
                output_language = gr.Radio(
                    choices=[
                        ("Auto (same as source) 自动（与源文档相同）", "auto"),
                        ("Chinese 中文输出", "chinese"),
                        ("English 英文输出", "english")
                    ],
                    value="auto",
                    label="Output Language 输出语言",
                    info="Choose the language for your summary regardless of the source document language"
                )

                summarize_button = gr.Button("🚀 Generate Summary 生成摘要", variant="primary", size="lg")

        gr.Markdown("### 📋 Summary Output 摘要输出")
        output_text = gr.Textbox(
            label="Summary 摘要",
            lines=20,
            max_lines=50,
            interactive=False,
            placeholder="Your enhanced document summary will appear here...\n您的文档摘要将显示在这里..."
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
            inputs=[file_input, summary_type, include_quotes, use_ocr, ocr_language, output_language],
            outputs=[output_text]
        )

        # Enhanced tips section
        gr.Markdown(
            """
            ### 💡 Quick Setup Guide 快速设置指南:

            **To Enable OCR (Required for scanned documents) 启用OCR（扫描文档必需）:**

            1. **Install Python packages 安装Python包:**
               ```bash
               pip install pytesseract pdf2image pillow opencv-python-headless
               ```

            2. **Install Tesseract OCR Software 安装Tesseract OCR软件:**
               - **Windows**: 
                 - Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
                 - During installation, select "Additional language data" → Chinese (Simplified & Traditional)

               - **Mac**: 
                 ```bash
                 brew install tesseract
                 brew install tesseract-lang  # Installs all language packs
                 ```

               - **Linux**: 
                 ```bash
                 sudo apt-get install tesseract-ocr
                 sudo apt-get install tesseract-ocr-chi-sim tesseract-ocr-chi-tra
                 ```

            3. **For better Chinese PDF support 更好的中文PDF支持:**
               ```bash
               pip install pdfplumber
               ```

            ### 🌐 Output Language Options 输出语言选项:
            - **Auto 自动**: Summary in the same language as the source document
            - **Chinese 中文**: Always output in Chinese, even for English documents
            - **English 英文**: Always output in English, even for Chinese documents

            ### ❓ Common Issues 常见问题:
            - **"OCR Not Available"**: Tesseract software not installed (not just the Python package)
            - **"Chinese OCR Not Ready"**: Chinese language packs not installed for Tesseract
            - **Corrupted text**: Try enabling OCR or installing pdfplumber
            """
        )

    return interface


# Requirements installation note
def print_requirements():
    print("""
    ====================================
    ENHANCED DOCUMENT SUMMARIZER WITH CHINESE OCR
    增强版文档摘要生成器（支持中文OCR）
    ====================================

    MINIMAL SETUP (for Chinese PDFs without OCR):
    pip install gradio langchain langchain-community PyPDF2 python-docx openai nltk pdfplumber

    FULL SETUP (with OCR support):
    pip install gradio langchain langchain-community PyPDF2 python-docx openai nltk pdfplumber pytesseract pdf2image pillow opencv-python-headless

    Then install Tesseract OCR software:
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - Mac: brew install tesseract tesseract-lang
    - Linux: sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra

    ====================================
    """)


if __name__ == "__main__":
    print_requirements()

    # Create and launch the interface
    interface = create_enhanced_gradio_interface()
    interface.launch(
        share=False,
        server_name="localhost",
        server_port=7860,
        show_error=True
    )