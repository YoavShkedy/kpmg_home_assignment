# 🏥 Medical Services ChatBot - Phase 2

## 🚀 Quick Start

**To run the system immediately:**

1. **Install dependencies:**
   ```bash
   cd part_2
   pip install -r requirements.txt
   ```

2. **Set up environment variables:**
   Create a `.env` file in the `part_2` directory with your Azure OpenAI credentials:
   ```bash
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   ```

3. **Run the system:**
   ```bash
   python run.py
   ```
   
   This will start both the FastAPI backend (port 8000) and Gradio frontend (port 7860).

4. **Access the chat interface:**
   Open http://localhost:7860 in your browser

---

## Overview
A sophisticated microservice-based ChatBot system for answering questions about medical services from Israeli health funds (Maccabi, Meuhedet, Clalit). The system uses LangGraph workflows, Azure OpenAI, FAISS vector search, and Gradio to provide personalized medical service information.

🏗️ Architecture
Frontend (Gradio) ←→ Backend (FastAPI) ←→ LangGraph Workflow
                                        ├── Azure OpenAI (LLM)
                                        ├── Azure OpenAI (Embeddings) 
                                        └── FAISS Vector Store
Key Features
Stateless Microservice: All user data managed client-side
LangGraph Workflows: Advanced conversation flow management
Vector Search: FAISS-powered semantic search over medical documents
Bilingual Support: Hebrew and English
Real-time Chat: Gradio-based responsive UI
Phase-based Interaction: Onboarding → Q&A workflow
📁 Project Structure
medical-chatbot/
├── app.py                      # FastAPI backend application
├── gradio_ui.py               # Gradio frontend interface
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── models/
│   └── schemas.py            # Pydantic data models
├── services/
│   ├── chat_service.py       # Main chat orchestration
│   ├── llm_service.py        # Azure OpenAI interactions
│   └── vector_service.py     # FAISS vector operations
├── langgraph/
│   └── workflow.py           # LangGraph workflow definition
├── scripts/
│   └── build_index.py        # Build FAISS index from HTML
├── vector_store/             # FAISS index storage (generated)
│   ├── faiss_index.bin
│   ├── documents.pkl
│   └── metadata.pkl
└── phase2_data/              # HTML knowledge base files
    ├── maccabi_services.html
    ├── meuhedet_services.html
    └── clalit_services.html
🚀 Setup Instructions
1. Prerequisites
Python 3.11+
Azure OpenAI access with deployed models:
gpt-4o (LLM)
text-embedding-3-small (Embeddings)
2. Environment Setup
Create a .env file in the project root:

bash
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
3. Install Dependencies
bash
pip install -r requirements.txt
4. Prepare Knowledge Base
Place your HTML knowledge base files in the phase2_data/ folder:

bash
mkdir -p phase2_data
# Copy your HTML files here
5. Build Vector Index
bash
python scripts/build_index.py
This will:

Process all HTML files in phase2_data/
Generate embeddings using Azure OpenAI
Create FAISS index in vector_store/
6. Start Backend Service
bash
python app.py
Backend will be available at: http://localhost:8000

7. Start Frontend Interface
In a new terminal:

bash
python gradio_ui.py
Frontend will be available at: http://localhost:7860

🔄 Workflow Overview
Phase 1: Onboarding
The LLM collects user information through natural conversation:

Personal Details: Name, ID number, gender, age
HMO Information: Health fund name, card number, membership tier
Confirmation: User reviews and confirms information
Phase 2: Q&A
Vector-enhanced question answering:

Retrieve: Search FAISS index for relevant medical service documents
Generate: LLM generates personalized answers based on user profile and retrieved content
🧪 Testing the System
1. Check API Status
bash
curl http://localhost:8000/
2. Check Vector Store
bash
curl http://localhost:8000/vector-store/stats
3. Test Chat API
bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "שלום, שמי דן כהן",
    "phase": "onboarding",
    "conversation_history": [],
    "user_profile": null
  }'
🎯 Usage Examples
Onboarding Flow
User: שלום, שמי דן כהן Bot: שלום דן! נחמד להכיר. איזה מספר זהות יש לך?

User: 123456789 Bot: תודה. כמה אתה בן?

[... continues until all information collected ...]

Q&A Flow
User: איזה בדיקות אני זכאי בחבילת הזהב של מכבי? Bot: בחבילת הזהב של מכבי אתה זכאי ל...

🛠️ Development
Adding New Services
Extend services/ with new service modules
Update models/schemas.py for new data structures
Modify langgraph/workflow.py for workflow changes
Custom LangGraph Nodes
python
def custom_node(state: WorkflowState) -> WorkflowState:
    # Your custom logic here
    return state

# Add to workflow
workflow.add_node("custom_node", custom_node)
Vector Store Management
python
from services.vector_service import VectorService

vector_service = VectorService()
results = vector_service.search("your query", k=5, hmo_filter="maccabi")
📊 Monitoring
Vector Store Statistics
The system provides real-time statistics:

Total indexed documents
Index dimension
HMO distribution
Conversation Tracking
All conversations are tracked client-side with:

Message timestamps
User profiles
Phase transitions
🔒 Security Considerations
Stateless Design: No server-side user data storage
Input Validation: Pydantic models ensure data integrity
Error Handling: Comprehensive exception management
CORS Configuration: Controlled frontend access
🚀 Deployment
Production Checklist
Environment Variables:
bash
export AZURE_OPENAI_API_KEY=your_key
export AZURE_OPENAI_ENDPOINT=your_endpoint
CORS Configuration: Update allowed origins in app.py
Process Management: Use gunicorn for FastAPI:
bash
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
Reverse Proxy: Configure nginx for production
Docker Deployment
dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python scripts/build_index.py

EXPOSE 8000 7860
CMD ["python", "app.py"]
🤝 Contributing
Fork the repository
Create feature branch: git checkout -b feature-name
Commit changes: git commit -am 'Add feature'
Push to branch: git push origin feature-name
Submit pull request
📝 License
This project is developed for KPMG GenAI Developer Assessment.

🆘 Troubleshooting
Common Issues
Vector Store Not Loading:
bash
python scripts/build_index.py
API Connection Error:
Check .env file configuration
Verify Azure OpenAI credentials
Ensure models are deployed
Empty Search Results:
Verify HTML files in phase2_data/
Check vector store statistics
Rebuild index if necessary
Logs and Debugging
Enable debug logging:

python
import logging
logging.basicConfig(level=logging.DEBUG)
Built with ❤️ using LangGraph, FastAPI, and Gradio

