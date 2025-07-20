from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import List, Dict, Any
from dotenv import load_dotenv

# Import the correct workflow and dependencies
from workflow.workflow import Workflow, WorkflowState
from models.schemas import ChatRequest, ChatResponse, ChatMessage, UserProfile
from services.vector_service import VectorService
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, BaseMessage

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Medical Services ChatBot API",
    description="Microservice-based ChatBot for Medical Services Q&A",
    version="1.0.0"
)

# Add CORS middleware to allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatService:
    """Simple chat service to handle basic functionality"""
    def __init__(self):
        pass
    
    def get_welcome_message(self) -> str:
        """Get welcome message"""
        return "Hi there! I am the HMO services chatbot. I would be happy to help you with questions about your HMO services. I can answer in both Hebrew and English. Can you please tell me your name?"

# Initialize services
try:
    # Initialize Azure OpenAI
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-12-01-preview",
        temperature=0
    )
    
    # Initialize vector service
    vector_service = VectorService("indexes")
    
    # Initialize workflow
    workflow_instance = Workflow(llm=llm, vector_service=vector_service)
    compiled_workflow = workflow_instance.build_workflow()
    
    # Initialize simple chat service
    chat_service = ChatService()
    
    print("✅ All services initialized successfully")
    
except Exception as e:
    print(f"❌ Error initializing services: {e}")
    raise

def convert_chat_history_to_langchain_messages(chat_history: List[ChatMessage]) -> List[BaseMessage]:
    """Convert ChatMessage list to LangChain message format"""
    messages = []
    for msg in chat_history:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
    return messages

def extract_response_from_workflow_result(final_state: Dict[str, Any]) -> tuple[str, UserProfile | None]:
    """Extract the response message and user profile from workflow result"""
    response_message = ""
    user_profile = None
    
    # Get the last assistant message from the final state
    for node_name, node_update in final_state.items():
        if "messages" in node_update:
            messages = node_update["messages"]
            # Find the last AI message
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content') and isinstance(msg.content, str):
                    response_message = msg.content
                    break
            if response_message:
                break
        
        # Get user profile if available
        if "user_profile" in node_update and node_update["user_profile"]:
            user_profile = node_update["user_profile"]
    
    return response_message, user_profile

def determine_phase(user_profile: UserProfile | None, response_message: str) -> str:
    """Determine the current phase based on user profile and response"""
    if user_profile is not None:
        return "qa"
    else:
        return "onboarding"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Medical Services ChatBot API is running"}

@app.get("/welcome")
async def get_welcome_message():
    """Get initial welcome message"""
    return {"message": chat_service.get_welcome_message()}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint - stateless conversation handling"""
    try:
        # Validate request
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Convert chat history to LangChain message format
        langchain_messages = convert_chat_history_to_langchain_messages(request.conversation_history)
        
        # Add the current user message
        langchain_messages.append(HumanMessage(content=request.message))
        
        # Prepare initial state for workflow
        initial_state: WorkflowState = {
            "messages": langchain_messages,
            "user_profile": request.user_profile
        }
        
        # Run the workflow
        final_state = None
        try:
            for chunk in compiled_workflow.stream(
                initial_state,
                config={"recursion_limit": 50}
            ):
                final_state = chunk
        except Exception as workflow_error:
            print(f"Workflow error: {workflow_error}")
            raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(workflow_error)}")
        
        if not final_state:
            raise HTTPException(status_code=500, detail="No response from workflow")
        
        # Extract response and user profile from workflow result
        response_message, updated_user_profile = extract_response_from_workflow_result(final_state)
        
        if not response_message:
            response_message = "מצטער, אני לא הצלחתי לעבד את הבקשה שלך. אנא נסה שוב."
        
        # Determine current phase
        current_phase = determine_phase(updated_user_profile, response_message)
        
        # Check if confirmation is required (when user profile is extracted but we're still in onboarding)
        requires_confirmation = (updated_user_profile is not None and 
                               current_phase == "qa" and 
                               request.user_profile is None)
        
        return ChatResponse(
            message=response_message,
            user_profile=updated_user_profile,
            phase=current_phase,
            requires_confirmation=requires_confirmation
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error processing chat request: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/vector-store/stats")
async def get_vector_store_stats():
    """Get vector store statistics"""
    try:
        stats = vector_service.get_stats()
        return stats
    except Exception as e:
        print(f"Error getting vector store stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving vector store statistics")

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )