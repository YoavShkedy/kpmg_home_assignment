# KPMG Assignment - Israeli Healthcare Document Processing

This repository contains a two-part assignment focused on AI-powered document processing and conversational systems for Israeli healthcare and insurance services.

## 📋 Part 1: Israeli National Insurance Form Extractor

**Location**: `part_1/`

A Streamlit web application that extracts structured information from Israeli National Insurance Institute (ביטוח לאומי) forms using:
- **Azure Document Intelligence** for OCR text extraction
- **Azure OpenAI GPT-4** for intelligent field extraction
- **Comprehensive validation** for data quality assurance

**Features**: Multi-format support (PDF, images), bilingual processing (Hebrew/English), JSON export

➡️ **[See detailed setup instructions](part_1/README.md)**

## 🏥 Part 2: Medical Services ChatBot

**Location**: `part_2/`

A sophisticated microservice-based chatbot system for answering questions about Israeli health fund services using:
- **LangGraph workflows** for conversation management
- **FAISS vector search** over medical service documents
- **Azure OpenAI** for embeddings and chat completion
- **Gradio frontend** with FastAPI backend

**Features**: Stateless design, bilingual support, personalized responses based on user profile and HMO membership

➡️ **[See detailed setup instructions](part_2/README.md)**

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (Part 1) / Python 3.11+ (Part 2)
- Azure Document Intelligence service
- Azure OpenAI service with GPT-4 and embedding models

### Running Each Part

**Part 1 - Form Extractor**:
```bash
cd part_1
pip install -r requirements.txt
# Configure .env file with Azure credentials
streamlit run app.py
```

**Part 2 - ChatBot**:
```bash
cd part_2
pip install -r requirements.txt
# Configure .env file with Azure credentials
python run.py  # Starts both backend and frontend
```

## 📁 Project Structure

```
kpmg_assignment/
├── part_1/                    # Form extraction application
│   ├── app.py                # Streamlit web interface
│   ├── field_extraction.py   # Azure OpenAI field extraction
│   ├── ocr.py               # Azure Document Intelligence OCR
│   └── README.md            # Detailed setup instructions
├── part_2/                   # ChatBot system
│   ├── app.py               # FastAPI backend
│   ├── gradio_ui.py         # Gradio frontend
│   ├── workflow/            # LangGraph conversation workflows
│   ├── services/            # Vector search and LLM services
│   └── README.md            # Detailed setup instructions
├── phase1_data/             # Sample PDF documents for testing
└── README.md               # This overview file
```

## 🔧 Technologies Used

- **Azure Document Intelligence** - OCR and document analysis
- **Azure OpenAI** - GPT-4 for text processing and embeddings
- **LangGraph** - Advanced conversation workflow management
- **FAISS** - Vector similarity search
- **Streamlit** - Web interface for form extraction
- **Gradio** - Chat interface
- **FastAPI** - REST API backend
- **Pydantic** - Data validation and schema management

## 📖 Documentation

Each part contains comprehensive documentation:
- Setup and configuration instructions
- Usage guides with examples
- Troubleshooting common issues
- Development guidelines

For detailed information about each component, please refer to the individual README files in the respective directories.
