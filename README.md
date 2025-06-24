# 📚 AI Document Summarizer with OCR Support | AI文档摘要生成器（支持OCR）

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://www.gradio.app/)
[![DeepSeek](https://img.shields.io/badge/API-DeepSeek-purple.svg)](https://www.deepseek.com/)

An intelligent document summarization tool powered by DeepSeek API with OCR capabilities for scanned documents. Supports both Chinese and English languages with a bilingual interface.

一个由DeepSeek API驱动的智能文档摘要工具，具备OCR功能，可处理扫描文档。支持中英文双语界面。

[English](#english) | [中文](#chinese)

</div>

---

## 🌟 Features | 功能特点

### Core Features | 核心功能
- 📄 **Multi-format Support** | 支持多种格式: PDF, DOCX, TXT, PNG, JPG, JPEG, TIFF, BMP, GIF
- 🔍 **OCR Capabilities** | OCR功能: Process scanned documents and images | 处理扫描文档和图片
- 🌐 **Bilingual Support** | 双语支持: Chinese and English interface & output | 中英文界面和输出
- 📊 **Multiple Summary Types** | 多种摘要类型:
  - Concise summaries | 简洁摘要
  - Detailed summaries | 详细摘要
  - Bullet points | 要点总结
  - Key insights | 关键见解
  - Chapter-wise summaries | 章节式摘要

### Performance Features | 性能特点
- ⚡ **High-Speed Processing** | 高速处理: Parallel OCR with timeout protection | 带超时保护的并行OCR
- 💾 **Smart Caching** | 智能缓存: Instant reprocessing of cached documents | 缓存文档即时处理
- 🔄 **Streaming Responses** | 流式响应: Real-time feedback during processing | 处理过程实时反馈
- 📈 **Progress Tracking** | 进度跟踪: Visual progress indicators | 可视化进度指示
- ⏹️ **Cancellable Operations** | 可取消操作: Stop long-running processes | 停止长时间运行的进程
- 🎯 **Quality Settings** | 质量设置: Balance between speed and accuracy | 速度与准确度平衡

## 🖼️ Screenshots | 截图

<div align="center">
<img src="https://github.com/WWIIITT/AI_PDF_summarizer/images/main-interface1.png" alt="Main Interface" width="800"/>
<img src="https://github.com/WWIIITT/AI_PDF_summarizer/images/main-interface2.png" alt="Main Interface" width="800"/>
<p><em>Main Interface | 主界面</em></p>
</div>

## 🚀 Quick Start | 快速开始

### Prerequisites | 前置要求

- Python 3.8 or higher | Python 3.8或更高版本
- DeepSeek API Key | DeepSeek API密钥
- Tesseract OCR (for OCR features) | Tesseract OCR（用于OCR功能）

### Installation | 安装

1. **Clone the repository | 克隆仓库**
   ```bash
   git clone https://github.com/yourusername/AI_PDF_summarizer.git
   cd AI_PDF_summarizer
   ```

2. **Create virtual environment | 创建虚拟环境**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies | 安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR | 安装Tesseract OCR**
   
   **Windows:**
   - Download from: https://github.com/UB-Mannheim/tesseract/wiki
   - Add to PATH or update the path in the code
   
   **macOS:**
   ```bash
   brew install tesseract
   brew install tesseract-lang  # For additional languages
   ```
   
   **Linux:**
   ```bash
   sudo apt-get install tesseract-ocr
   sudo apt-get install tesseract-ocr-chi-sim  # For Chinese
   ```

5. **Run the application | 运行应用**
   ```bash
   python pdf_summarizer.py
   ```

## 📋 Requirements | 依赖要求

Create a `requirements.txt` file with:

```txt
gradio>=4.0.0
langchain>=0.1.0
langchain-community>=0.0.10
PyPDF2>=3.0.0
python-docx>=0.8.11
openai>=1.0.0
nltk>=3.8
pdfplumber>=0.9.0
pytesseract>=0.3.10
pdf2image>=1.16.3
pillow>=10.0.0
opencv-python-headless>=4.8.0
psutil>=5.9.0
numpy>=1.24.0
```

## 💻 Usage | 使用方法

### 1. Set API Key | 设置API密钥
- Enter your DeepSeek API key in the interface | 在界面中输入您的DeepSeek API密钥
- Click "Set API Key" | 点击"设置API密钥"

### 2. Upload Document | 上传文档
- Click "Upload Document" | 点击"上传文档"
- Select your file | 选择您的文件
- Use "Quick Analysis" for document preview | 使用"快速分析"预览文档

### 3. Configure Settings | 配置设置
- **Summary Type** | 摘要类型: Choose your preferred format | 选择您偏好的格式
- **Processing Quality** | 处理质量:
  - Fast (100 DPI): Quick results | 快速（100 DPI）：快速结果
  - Balanced (150 DPI): Optimal balance | 平衡（150 DPI）：最佳平衡
  - High Quality (200 DPI): Best accuracy | 高质量（200 DPI）：最佳准确度
- **OCR Settings** | OCR设置:
  - Enable/Disable OCR | 启用/禁用OCR
  - Set maximum OCR pages | 设置最大OCR页数
  - Choose OCR language | 选择OCR语言

### 4. Generate Summary | 生成摘要
- Click "Generate Summary" | 点击"生成摘要"
- Monitor progress in real-time | 实时监控进度
- Cancel anytime if needed | 需要时随时取消

## ⚙️ Configuration | 配置

### API Configuration | API配置
```python
# DeepSeek API settings
model = 'deepseek-chat'
api_base = 'https://api.deepseek.com'
max_tokens = 2048
temperature = 0.3
```

### OCR Configuration | OCR配置
```python
# Tesseract paths for Windows
tesseract_paths = [
    r"D:\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]
```

### Performance Settings | 性能设置
```python
# Chunk settings
chunk_size = 4000
chunk_overlap = 400

# OCR settings
max_ocr_pages = 20  # Default maximum pages for OCR
ocr_timeout = 30    # Timeout per page in seconds

# Cache settings
cache_dir = tempfile.gettempdir()
```

## 🚄 Performance Optimization | 性能优化

### For Large PDFs | 处理大型PDF
- Limit OCR pages to 10-20 for faster processing | 限制OCR页数到10-20页以加快处理
- Use "Fast" quality for initial testing | 使用"快速"质量进行初始测试
- Enable caching for repeated processing | 启用缓存以重复处理

### OCR Best Practices | OCR最佳实践
- Disable OCR if text is already selectable | 如果文本已可选择，禁用OCR
- Use appropriate language settings | 使用适当的语言设置
- Process in batches for very large documents | 对超大文档分批处理

### Memory Management | 内存管理
- Automatic garbage collection after batch processing | 批处理后自动垃圾回收
- Configurable batch sizes | 可配置的批次大小
- Memory-efficient image processing | 内存高效的图像处理

## 🛠️ Technical Details | 技术细节

### Architecture | 架构
- **Frontend**: Gradio web interface | Gradio网页界面
- **Backend**: LangChain + DeepSeek API | LangChain + DeepSeek API
- **OCR Engine**: Tesseract OCR | Tesseract OCR引擎
- **Caching**: File-based caching system | 基于文件的缓存系统

### Key Components | 关键组件
1. **Document Processor**: Handles multiple file formats | 处理多种文件格式
2. **OCR Module**: Processes scanned documents | 处理扫描文档
3. **Text Splitter**: Manages large documents | 管理大型文档
4. **Summarizer**: Generates various summary types | 生成各种摘要类型
5. **Cache Manager**: Improves performance | 提升性能

## 🤝 Contributing | 贡献

Contributions are welcome! Please feel free to submit a Pull Request.

欢迎贡献！请随时提交Pull Request。

1. Fork the repository | Fork仓库
2. Create your feature branch | 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. Commit your changes | 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch | 推送到分支 (`git push origin feature/AmazingFeature`)
5. Open a Pull Request | 开启Pull Request


## 🙏 Acknowledgments | 致谢

- [DeepSeek](https://www.deepseek.com/) for providing the API | 提供API
- [LangChain](https://github.com/langchain-ai/langchain) for the framework | 提供框架
- [Gradio](https://www.gradio.app/) for the web interface | 提供网页界面
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for OCR capabilities | 提供OCR功能


---

<div align="center">
Made with ❤️ by HO Cheuk Ting

如果觉得有用，请给个⭐️ Star！| If you find this useful, please give it a ⭐️ Star!
</div>
