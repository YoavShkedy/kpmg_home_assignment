from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import re

class UserProfile(BaseModel):
    first_name: str = Field(..., description="User's first name")
    last_name: str = Field(..., description="User's last name") 
    national_id: str = Field(..., description="9-digit national ID number")
    gender: str = Field(..., description="Gender (male|female / זכר|נקבה)")
    date_of_birth: str = Field(..., description="Date of birth in DD/MM/YYYY format")
    hmo: str = Field(..., description="HMO name (Clalit|Maccabi|Meuhedet / כללית|מכבי|מאוחדת)")
    insurance_tier: str = Field(..., description="Insurance membership tier (gold|silver|bronze / זהב|כסף|ארד)")

class FieldExtraction(BaseModel):
    """Schema for extracting individual field values"""
    field: str = Field(..., description="The field name being extracted")
    value: str = Field(..., description="The extracted value for the field")

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    user_profile: Optional[UserProfile] = Field(None, description="User profile if available")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Chat history")
    phase: str = Field(default="onboarding", description="Current phase: onboarding or qa")

class ChatResponse(BaseModel):
    message: str = Field(..., description="Assistant's response")
    user_profile: Optional[UserProfile] = Field(None, description="Updated user profile")
    phase: str = Field(..., description="Current phase")
    requires_confirmation: bool = Field(default=False, description="Whether user needs to confirm profile")

class RetrievalResult(BaseModel):
    content: str = Field(..., description="Retrieved document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    score: float = Field(..., description="Similarity score")

class WorkflowState(BaseModel):
    message: str
    user_profile: Optional[UserProfile] = None
    conversation_history: List[ChatMessage] = Field(default_factory=list)
    phase: str = "onboarding"
    retrieved_docs: List[RetrievalResult] = Field(default_factory=list)
    response: str = ""
    requires_confirmation: bool = False
    
    # Information collection specific fields
    collection_complete: bool = False
    extraction_attempted: bool = False
    extraction_complete: bool = False