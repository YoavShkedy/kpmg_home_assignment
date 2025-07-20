#!/usr/bin/env python3
"""
Gradio frontend for Medical Services ChatBot
"""

import gradio as gr
import httpx
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import asyncio

# API Configuration
API_BASE_URL = "http://localhost:8000"

class ChatState:
    """Client-side state management"""
    
    def __init__(self):
        self.conversation_history = []
        self.user_profile = None
        self.phase = "onboarding"
        self.requires_confirmation = False
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def set_user_profile(self, profile_data: Dict[str, Any]):
        """Set user profile"""
        self.user_profile = profile_data
    
    def reset(self):
        """Reset state"""
        self.__init__()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for API calls"""
        return {
            "conversation_history": self.conversation_history,
            "user_profile": self.user_profile,
            "phase": self.phase,
            "requires_confirmation": self.requires_confirmation
        }

# Global state instance
chat_state = ChatState()

async def get_welcome_message() -> str:
    """Get welcome message from API"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{API_BASE_URL}/welcome")
            if response.status_code == 200:
                return response.json()["message"]
            else:
                return "Welcome to Medical Services ChatBot!"
    except Exception as e:
        print(f"Error getting welcome message: {e}")
        return "Welcome to Medical Services ChatBot!"

async def send_message_to_api(message: str) -> Tuple[str, str, bool]:
    """Send message to API and return response"""
    try:
        request_data = {
            "message": message,
            "user_profile": chat_state.user_profile,
            "conversation_history": chat_state.conversation_history,
            "phase": chat_state.phase
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{API_BASE_URL}/chat",
                json=request_data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["message"], result["phase"], result.get("requires_confirmation", False)
            else:
                error_msg = f"API Error: {response.status_code}"
                try:
                    error_detail = response.json().get("detail", "Unknown error")
                    error_msg += f" - {error_detail}"
                except:
                    pass
                return error_msg, chat_state.phase, False
                
    except httpx.TimeoutException:
        return "Request timed out. Please try again.", chat_state.phase, False
    except Exception as e:
        print(f"Error sending message to API: {e}")
        return f"Error: {str(e)}", chat_state.phase, False

def format_chat_history() -> List[Tuple[str, str]]:
    """Format conversation history for Gradio chatbot"""
    formatted_history = []
    
    for msg in chat_state.conversation_history:
        if msg["role"] == "user":
            if formatted_history and formatted_history[-1][1] is None:
                # Update the last user message
                formatted_history[-1] = (msg["content"], None)
            else:
                # Add new user message
                formatted_history.append((msg["content"], None))
        elif msg["role"] == "assistant":
            if formatted_history:
                # Add assistant response to the last user message
                formatted_history[-1] = (formatted_history[-1][0], msg["content"])
            else:
                # Shouldn't happen, but handle gracefully
                formatted_history.append(("", msg["content"]))
    
    return formatted_history

async def process_user_message(message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
    """Process user message and return updated history"""
    if not message.strip():
        return history, ""
    
    # Add user message to state
    chat_state.add_message("user", message)
    
    # Send to API
    response_message, new_phase, requires_confirmation = await send_message_to_api(message)
    
    # Update state
    chat_state.phase = new_phase
    chat_state.requires_confirmation = requires_confirmation
    
    # Add assistant response to state
    chat_state.add_message("assistant", response_message)
    
    # Format updated history
    updated_history = format_chat_history()
    
    return updated_history, ""

def reset_conversation():
    """Reset the conversation"""
    chat_state.reset()
    return [], ""

def get_current_phase_info() -> str:
    """Get information about current phase"""
    if chat_state.phase == "onboarding":
        profile_status = "❌ Not collected" if not chat_state.user_profile else "✅ Collected"
        return f"**Phase:** Onboarding\n**Profile:** {profile_status}"
    elif chat_state.phase == "qa":
        if chat_state.user_profile:
            return f"**Phase:** Q&A\n**User:** {chat_state.user_profile.get('first_name', '')} {chat_state.user_profile.get('last_name', '')}\n**HMO:** {chat_state.user_profile.get('hmo', '')}"
        else:
            return "**Phase:** Q&A\n**Profile:** ❌ Missing"
    else:
        return f"**Phase:** {chat_state.phase}"

async def get_api_status() -> str:
    """Check API status"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE_URL}/")
            if response.status_code == 200:
                return "🟢 API Connected"
            else:
                return f"🔴 API Error: {response.status_code}"
    except Exception as e:
        return f"🔴 API Offline: {str(e)}"

async def get_vector_store_status() -> str:
    """Check vector store status"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_BASE_URL}/vector-store/stats")
            if response.status_code == 200:
                stats = response.json()
                if stats["status"] == "loaded":
                    return f"🟢 Vector Store: {stats['total_documents']} documents"
                else:
                    return "🔴 Vector Store: Not loaded"
            else:
                return f"🔴 Vector Store Error: {response.status_code}"
    except Exception as e:
        return f"🔴 Vector Store Offline: {str(e)}"

def create_gradio_interface():
    """Create and configure Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .chat-container { max-height: 600px; overflow-y: auto; }
    .status-panel { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
    .phase-info { background-color: #e6f3ff; padding: 10px; border-radius: 5px; }
    """
    
    with gr.Blocks(css=css, title="Medical Services ChatBot") as demo:
        
        gr.Markdown("""
        # 🏥 Medical Services ChatBot
        
        **Advanced AI Assistant for Israeli Health Funds**
        
        This chatbot helps you with questions about medical services from Israeli health funds (Maccabi, Meuhedet, Clalit).
        The system uses advanced LangGraph workflows and vector search to provide personalized responses.
        """)
        
        with gr.Row():
            with gr.Column(scale=3):
                # Main chat interface
                chatbot = gr.Chatbot(
                    label="Chat with Medical Services Bot",
                    height=500,
                    elem_classes=["chat-container"]
                )
                
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Type your message here...",
                    lines=2
                )
                
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Reset Conversation", variant="secondary")
            
            with gr.Column(scale=1):
                # Status panel
                gr.Markdown("### 📊 Status Panel")
                
                phase_info = gr.Markdown(
                    value=get_current_phase_info(),
                    elem_classes=["phase-info"]
                )
                
                api_status = gr.Markdown(
                    value="🔄 Checking API...",
                    elem_classes=["status-panel"]
                )
                
                vector_status = gr.Markdown(
                    value="🔄 Checking Vector Store...",
                    elem_classes=["status-panel"]
                )
                
                refresh_btn = gr.Button("Refresh Status", size="sm")
        
        # Event handlers
        async def handle_send(message, history):
            """Handle send button click"""
            result = await process_user_message(message, history)
            # Also update phase info
            phase_info_text = get_current_phase_info()
            return result + (phase_info_text,)
        
        async def handle_refresh():
            """Handle refresh status button"""
            api_stat = await get_api_status()
            vector_stat = await get_vector_store_status()
            phase_info_text = get_current_phase_info()
            return api_stat, vector_stat, phase_info_text
        
        def handle_clear():
            """Handle clear button"""
            reset_conversation()
            phase_info_text = get_current_phase_info()
            return [], "", phase_info_text
        
        # Wire up events
        send_btn.click(
            fn=handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, phase_info]
        )
        
        msg_input.submit(
            fn=handle_send,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, phase_info]
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot, msg_input, phase_info]
        )
        
        refresh_btn.click(
            fn=handle_refresh,
            outputs=[api_status, vector_status, phase_info]
        )
        
        # Load initial status on startup
        demo.load(
            fn=handle_refresh,
            outputs=[api_status, vector_status, phase_info]
        )
        
        # Add welcome message on startup
        async def load_welcome():
            welcome_msg = await get_welcome_message()
            chat_state.add_message("assistant", welcome_msg)
            return format_chat_history()
        
        demo.load(
            fn=load_welcome,
            outputs=[chatbot]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Instructions:**
        1. **Onboarding Phase**: Provide your personal information when prompted
        2. **Q&A Phase**: Ask questions about medical services after profile confirmation
        
        **Supported Languages**: Hebrew (עברית) and English
        
        **Supported HMOs**: Maccabi (מכבי), Meuhedet (מאוחדת), Clalit (כללית)
        """)
    
    return demo

def main():
    """Main function to run Gradio interface"""
    print("Starting Medical Services ChatBot Gradio Interface...")
    
    demo = create_gradio_interface()
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )

if __name__ == "__main__":
    main()