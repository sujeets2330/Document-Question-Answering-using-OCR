 # 🧠 Document Question Answering using OCR

A full-stack AI system that extracts text from scanned documents using OCR and answers questions based on the layout and content using Transformer-based models like LayoutLMv3. Built with Flask, PyTorch, and Hugging Face.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-WebApp-lightgrey)
![LayoutLMv3](https://img.shields.io/badge/Model-LayoutLMv3-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🔍 Project Overview

This project merges **OCR** with **Document Question Answering (DocVQA)** to enable smart understanding of scanned or photographed documents. Users can upload an image, ask a question about its content (like invoice number, date, amount), and the system gives an accurate answer by processing both text and layout.

---

## 🎯 Objectives

- Extract structured text data from documents using OCR
- Use layout-aware models (LayoutLMv3) to comprehend document structure
- Answer natural language questions based on the document content
- Deploy with a clean, responsive web interface using Flask
- Provide fully logged and structured outputs

---

## 💡 Real-World Applications

- 🏛 Government records automation
- 🧾 Invoice & receipt parsing
- 🏥 Medical report extraction
- 📚 Educational transcripts analysis
- 📑 Enterprise document summarization

---

## 🛠️ Tech Stack

**Languages & Frameworks**
- Python 3.8+
- Flask (Web UI & Backend)

**OCR Engine**
- Tesseract / PaddleOCR

**Transformers**
- Hugging Face Transformers
- LayoutLMv3 / LayoutLM

**Other Tools**
- PyTorch
- OpenCV
- JSON / Logging

---



---

## 🚀 Features

- 📷 **OCR-Powered Document Input**
- 🧠 **Layout-Aware QA with LayoutLMv3**
- 🖥 **Clean Flask-based Web Interface**
- 🔄 **Fully Automatic Pipeline**
- 📂 **Structured Logs and Result Files**
- ✅ **Works with Invoices, Bills, Receipts, Forms, and more**

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
 
git clone https://github.com/yourusername/docqa-ocr.git
cd docqa-ocr

### 2. Create a Virtual Environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

### 3. Install Requirements
pip install -r requirements.txt

### 4. Download & Load Model Weights
The model is downloaded automatically via Hugging Face when first used.

 ###  Example :

from transformers import LayoutLMv3Processor, LayoutLMv3ForQuestionAnswering

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForQuestionAnswering.from_pretrained("microsoft/layoutlmv3-base")

###  5. Run the Application
python app.py 

Access the app at: http://127.0.0.1:5000

 ###  🖼 Architecture Diagram
        [User Upload]
             ↓
    ┌──────────────────────┐
    │   OCR (Tesseract)    │ ← Text + Layout
    └──────────────────────┘
             ↓
    ┌──────────────────────┐
    │ LayoutLMv3 Processor │ ← Tokenization + Bounding Boxes
    └──────────────────────┘
             ↓
    ┌──────────────────────┐
    │ LayoutLMv3 Model     │ ← QA Task
    └──────────────────────┘
             ↓
       [Answer Displayed]
       
### 📬 Contact
👤 Sujeet M A

🔗 GitHub: https://github.com/sujeets2330

📧 Email: sujeetmalagundi999@gmail.com


### 🙏 Acknowledgements
Hugging Face Transformers

Tesseract OCR

Microsoft LayoutLMv3

PaddleOCR
