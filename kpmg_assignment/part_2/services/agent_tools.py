from services.vector_service import VectorService
from typing import Annotated, Optional
from langchain_core.tools import tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import BaseMessage
from typing import List
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from models.schemas import UserProfile
import os
import json

# Define a message state type for the handoff tool
class MessagesState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Define the workflow state type to match the one in workflow.py
class WorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: Optional[UserProfile]

def load_prompt_from_file(filename: str) -> str:
    """Load prompt content from a file"""
    # Get the directory of this script and go up one level to find prompts
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prompts_dir = os.path.join(os.path.dirname(script_dir), "prompts")
    prompt_path = os.path.join(prompts_dir, filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

def create_extract_user_info_tool(llm: AzureChatOpenAI):
    """Create the extract_user_info tool"""
    
    @tool
    def extract_user_info(
        conversation_history: Annotated[str, "The conversation history to extract user information from"]
    ) -> str:
        """
        Extract user profile from conversation history.

        Args:
            conversation_history (str): The conversation history to extract user information from

        Returns:
            str: JSON string containing the extracted user profile
        """
        try:
            if llm is None:
                raise ValueError("LLM is required for user information extraction")
                
            # Load the extraction prompt
            info_extraction_prompt_content = load_prompt_from_file("info_extraction.txt")
            
            # Create prompt template
            extraction_prompt = ChatPromptTemplate.from_messages([
                ("system", info_extraction_prompt_content),
                ("user", "<chat_history>{conversation_history}</chat_history>")
            ])
            
            # Create LLM with structured output for UserProfile
            structured_llm = llm.with_structured_output(UserProfile)
            
            # Chain the prompt with the structured LLM
            extraction_chain = extraction_prompt | structured_llm
            
            # Extract the user profile from the conversation
            extracted_result = extraction_chain.invoke({"conversation_history": conversation_history})

            # Convert to dict and then to JSON string
            if isinstance(extracted_result, dict):
                result_dict = extracted_result
            elif hasattr(extracted_result, 'model_dump'):
                result_dict = extracted_result.model_dump()
            elif hasattr(extracted_result, 'dict'):
                result_dict = extracted_result.dict()
            else:
                result_dict = extracted_result.__dict__
            
            return json.dumps(result_dict)
            
        except Exception as e:
            # If extraction fails, raise an error
            raise ValueError(f"Error extracting user information: {str(e)}")
    
    return extract_user_info

def create_search_info_tool(vector_service: VectorService):
    """Create the search_info tool"""
    
    @tool
    def search_info(
        question: Annotated[str, "Any question about HMO services that the information to answer it is not available in chat history"]
    ) -> str:
        """
        Use vector similarity search to retrieve HMO-services-related information from the knowledge base.

        Args:
            question (str): Natural language question from the user.

        Returns:
            str: Relevant document excerpts from the HMO services knowledge base.
        """
        results = vector_service.search(question)
        
        # Convert RetrievalResult list to string format
        if not results:
            return "No relevant information found in the knowledge base."
        
        formatted_results = []
        for i, result in enumerate(results[:3], 1):  # Take top 3 results
            formatted_results.append(f"Result {i}:\n{result.content}\n")
        
        return "\n".join(formatted_results)
    
    return search_info

class Tools:
    """
    Class that defines tools which the agents can use in their workflows.
    """
    # --- Constructor ---
    def __init__(self, vector_service: VectorService, llm: Optional[AzureChatOpenAI] = None):
        self.vector_service = vector_service
        self.llm = llm
        # Create the tools
        if llm is not None:
            self.extract_user_info = create_extract_user_info_tool(llm)
        else:
            raise ValueError("LLM is required for user information extraction")
        self.search_info = create_search_info_tool(vector_service)