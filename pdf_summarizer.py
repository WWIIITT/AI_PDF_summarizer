import gradio as gr
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import PyPDF2
import docx
import io
import warnings

warnings.filterwarnings('ignore')


class DocumentSummarizer:
    def __init__(self, api_key):
        self.llm = ChatOpenAI(
            model='deepseek-chat',
            openai_api_key=api_key,
            openai_api_base='https://api.deepseek.com',
            max_tokens=1024,
            temperature=0.3
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            length_function=len,
        )

    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"

    def extract_text_from_docx(self, file_path):
        """Extract text from Word document"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            return f"Error reading Word document: {str(e)}"

    def get_file_text(self, file_path):
        """Extract text based on file extension"""
        if file_path.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.docx', '.doc')):
            return self.extract_text_from_docx(file_path)
        else:
            return "Unsupported file format. Please upload a PDF or Word document."

    def summarize_text(self, text, summary_type="concise"):
        """Summarize the extracted text"""
        if not text or text.startswith("Error") or text.startswith("Unsupported"):
            return text

        # Create documents from text chunks
        chunks = self.text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        if not documents:
            return "No content found to summarize."

        # Define different summary prompts
        if summary_type == "concise":
            prompt_template = """Write a concise summary of the following text:

{text}

CONCISE SUMMARY:"""
        elif summary_type == "detailed":
            prompt_template = """Write a detailed summary of the following text, including key points and important details:

{text}

DETAILED SUMMARY:"""
        elif summary_type == "bullet_points":
            prompt_template = """Create a bullet-point summary of the following text, highlighting the main points:

{text}

BULLET POINT SUMMARY:"""
        else:
            prompt_template = """Summarize the following text:

{text}

SUMMARY:"""

        try:
            # Use map-reduce chain for longer documents
            if len(documents) > 1:
                chain = load_summarize_chain(
                    self.llm,
                    chain_type="map_reduce",
                    verbose=False
                )
                summary = chain.run(documents)
            else:
                # For shorter documents, use direct summarization
                messages = [
                    SystemMessage(content="You are a helpful assistant that creates clear and informative summaries."),
                    HumanMessage(content=prompt_template.format(text=text))
                ]
                response = self.llm(messages)
                summary = response.content

            return summary
        except Exception as e:
            return f"Error during summarization: {str(e)}"


def create_gradio_interface():
    """Create the Gradio interface"""

    # Initialize with placeholder - user will need to set their API key
    summarizer = None

    def set_api_key(api_key):
        """Set the API key and initialize the summarizer"""
        global summarizer
        if api_key.strip():
            try:
                summarizer = DocumentSummarizer(api_key.strip())
                return "✅ API Key set successfully!"
            except Exception as e:
                return f"❌ Error setting API key: {str(e)}"
        else:
            return "❌ Please enter a valid API key"

    def process_document(file, summary_type, progress=gr.Progress()):
        """Process the uploaded document and return summary"""
        global summarizer

        if summarizer is None:
            return "❌ Please set your DeepSeek API key first!"

        if file is None:
            return "❌ Please upload a file!"

        try:
            progress(0.2, desc="Extracting text from document...")

            # Extract text from the uploaded file
            text = summarizer.get_file_text(file.name)

            if text.startswith("Error") or text.startswith("Unsupported"):
                return text

            progress(0.6, desc="Generating summary...")

            # Generate summary
            summary = summarizer.summarize_text(text, summary_type)

            progress(1.0, desc="Complete!")

            return summary

        except Exception as e:
            return f"❌ Error processing document: {str(e)}"

    # Create the Gradio interface
    with gr.Blocks(title="Document Summarizer", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 📄 Document Summarizer

            Upload PDF or Word documents and get AI-powered summaries using DeepSeek.

            **Supported formats:** PDF (.pdf), Word (.docx, .doc)
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

            with gr.Column(scale=2):
                gr.Markdown("### 📤 Document Upload")
                file_input = gr.File(
                    label="Upload Document",
                    file_types=[".pdf", ".docx", ".doc"],
                    type="filepath"
                )

                summary_type = gr.Radio(
                    choices=["concise", "detailed", "bullet_points"],
                    value="concise",
                    label="Summary Type"
                )

                summarize_button = gr.Button("📝 Generate Summary", variant="primary", size="lg")

        gr.Markdown("### 📋 Summary Output")
        output_text = gr.Textbox(
            label="Summary",
            lines=15,
            max_lines=50,
            interactive=False,
            placeholder="Your document summary will appear here..."
        )

        # Event handlers
        api_key_button.click(
            fn=set_api_key,
            inputs=[api_key_input],
            outputs=[api_key_status]
        )

        summarize_button.click(
            fn=process_document,
            inputs=[file_input, summary_type],
            outputs=[output_text]
        )

        # Example section
        gr.Markdown(
            """
            ### 💡 Tips:
            - **Concise**: Brief overview of main points
            - **Detailed**: Comprehensive summary with key details
            - **Bullet Points**: Organized list format

            ### 🔧 Setup:
            1. Get your API key from [DeepSeek](https://platform.deepseek.com/)
            2. Enter your API key above
            3. Upload your document (PDF or Word)
            4. Choose summary type and generate!
            """
        )

    return interface


# Requirements installation note
def print_requirements():
    print("""
    To run this application, install the required packages:

    pip install gradio langchain langchain-community PyPDF2 python-docx openai

    You'll also need a DeepSeek API key from: https://platform.deepseek.com/
    """)


if __name__ == "__main__":
    print_requirements()

    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(
        share=False,  # Set to True if you want to create a public link
        server_name="localhost",
        server_port=7860,
        show_error=True
    )