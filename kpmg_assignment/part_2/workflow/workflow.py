import sys
import os
# Add the part_2 directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from typing import List, Optional
from langchain.schema import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from models.schemas import UserProfile
from services.vector_service import VectorService

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.constants import Send
from services.agent_tools import Tools
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from typing import Annotated, TypedDict
import json

# Load environment variables
load_dotenv()

# Define the state for the workflow - Updated to include user_profile
class WorkflowState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_profile: Optional[UserProfile]

class Workflow:
    """
    Class that defines the workflow of the chatbot.
    """
    def __init__(self, llm: AzureChatOpenAI, vector_service: VectorService):
        self.llm = llm
        self.vector_service = vector_service
        self.agents = self.create_agents()

    def _load_prompt_from_file(self, filename: str) -> str:
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

    def create_agents(self):
        """
        Create the agents for the workflow.
        """
        # Initialize tools with LLM for extraction tool
        tools = Tools(vector_service=self.vector_service, llm=self.llm)

        # --- Create info collection agent ---

        # Load the prompt from the file
        info_collection_prompt_content = self._load_prompt_from_file("info_collection.txt")
        
        # Create prompt template
        collector_prompt = ChatPromptTemplate.from_messages([
            ("system", info_collection_prompt_content),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Create collector agent with extraction tool
        collector_llm = self.llm.bind_tools([tools.extract_user_info])

        def get_messages_info(messages):
            return [SystemMessage(content=info_collection_prompt_content)] + messages

        def collector_agent(state):
            messages = get_messages_info(state["messages"])
            response = collector_llm.invoke(messages)
            return {"messages": [response]}

        # --- Create QA agent ---

        # Load the prompt from the file
        qa_prompt_content = self._load_prompt_from_file("qa.txt")
        
        # Create QA agent with search tool
        qa_llm = self.llm.bind_tools([tools.search_info])

        def get_messages_qa(messages):
            return [SystemMessage(content=qa_prompt_content)] + messages

        def qa_agent(state):
            messages = get_messages_qa(state["messages"])
            response = qa_llm.invoke(messages)
            return {"messages": [response]}
        
        return {
            "collector_agent": collector_agent,
            "qa_agent": qa_agent
        }

    def build_workflow(self):
        """
        Build the workflow of the chatbot.
        """

        def route_after_collector(state: WorkflowState):
            """Route after collector agent - check if extraction tool was called"""
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "add_tool_message"
            elif not isinstance(last_message, HumanMessage):
                return END
            return "collector_agent"

        def route_after_qa(state: WorkflowState):
            """Route after QA agent - check if search tool was called"""
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "handle_qa_tool"
            elif not isinstance(last_message, HumanMessage):
                return END
            return "qa_agent"

        # Create the workflow graph
        workflow = StateGraph(WorkflowState)
    
        # Add nodes
        workflow.add_node("collector_agent", self.agents["collector_agent"])
        workflow.add_node("qa_agent", self.agents["qa_agent"])
        
        # Add the tool message handler node for collector
        @workflow.add_node
        def add_tool_message(state: WorkflowState):
            """Add a tool message in response to the tool call and extract user profile"""
            last_message = state["messages"][-1]
            tool_call = last_message.tool_calls[0]
            
            # Execute the tool to get the result
            tools = Tools(vector_service=self.vector_service, llm=self.llm)
            try:
                # Format the conversation history for the tool
                conversation_text = "\n".join([
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                    for msg in state["messages"] 
                    if isinstance(msg, (HumanMessage, AIMessage)) and hasattr(msg, 'content')
                ])
                
                # Call the extraction tool
                extraction_result = tools.extract_user_info.invoke({"conversation_history": conversation_text})
                
                # Parse the result to create UserProfile
                user_profile_data = json.loads(extraction_result)
                user_profile = UserProfile(**user_profile_data)
                
                tool_result = "User information collected successfully! How can I help you with HMO services?"
                
            except Exception as e:
                print(f"Error in tool execution: {e}")
                user_profile = None
                tool_result = f"Error collecting user information: {str(e)}"
            
            return {
                "messages": [
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                ],
                "user_profile": user_profile
            }

        # Add the tool message handler node for QA
        @workflow.add_node
        def handle_qa_tool(state: WorkflowState):
            """Handle tool calls from QA agent"""
            last_message = state["messages"][-1]
            tool_call = last_message.tool_calls[0]
            
            # Execute the search tool
            tools = Tools(vector_service=self.vector_service, llm=self.llm)
            try:
                # Get the question from the tool call
                question = tool_call["args"]["question"]
                
                # Call the search tool
                search_result = tools.search_info.invoke({"question": question})
                
                tool_result = search_result
                
            except Exception as e:
                print(f"Error in search tool execution: {e}")
                tool_result = f"Error searching for information: {str(e)}"
            
            return {
                "messages": [
                    ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_call["id"],
                    )
                ]
            }
        
        # Start at the collector agent
        workflow.add_edge(START, "collector_agent")
        
        # Add conditional edges from collector agent
        workflow.add_conditional_edges(
            "collector_agent",
            route_after_collector,
            ["add_tool_message", "collector_agent", END]
        )
        
        # Add edge from tool message to QA agent
        workflow.add_edge("add_tool_message", "qa_agent")
        
        # Add conditional edges from QA agent
        workflow.add_conditional_edges(
            "qa_agent",
            route_after_qa,
            ["handle_qa_tool", "qa_agent", END]
        )
        
        # Add edge from QA tool handler back to QA agent
        workflow.add_edge("handle_qa_tool", "qa_agent")
        
        # Compile and return the workflow
        return workflow.compile()
    
if __name__ == "__main__":

    # --- Functions to Print Chatbot Flow ---

    from langchain_core.messages import convert_to_messages


    def pretty_print_message(message, indent=False):
        pretty_message = message.pretty_repr(html=True)
        if not indent:
            print(pretty_message)
            return

        indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
        print(indented)


    def pretty_print_messages(update, last_message=False):
        is_subgraph = False
        if isinstance(update, tuple):
            ns, update = update
            # skip parent graph updates in the printouts
            if len(ns) == 0:
                return

            graph_id = ns[-1].split(":")[0]
            print(f"Update from subgraph {graph_id}:")
            print("\n")
            is_subgraph = True

        for node_name, node_update in update.items():
            update_label = f"Update from node {node_name}:"
            if is_subgraph:
                update_label = "\t" + update_label

            print(update_label)
            print("\n")

            messages = convert_to_messages(node_update["messages"])
            if last_message:
                messages = messages[-1:]

            for m in messages:
                pretty_print_message(m, indent=is_subgraph)
            print("\n")
    
    # Load environment variables
    load_dotenv()

    # Initialize the workflow
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4o",
        api_version="2024-12-01-preview",
        temperature=0
    )

    # Initialize the vector service
    vector_service = VectorService()

    # Initialize the workflow
    workflow = Workflow(llm=llm, vector_service=vector_service)

    # Build the workflow
    compiled_workflow = workflow.build_workflow()

    # Save the workflow graph to a file
    with open("workflow.png", "wb") as f:
        f.write(compiled_workflow.get_graph().draw_mermaid_png())
    
    # Initialize conversation history
    conversation_messages = []

    # Print chat history like a user would see it
    while True:
        try:
            user_input = input("\n👤 You: ")
            if not user_input.strip():
                continue

            if user_input.lower() == "exit":
                print("\n\n👋 Thank you and goodbye!")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Thank you and goodbye!")
            break

        # Add user message to conversation history
        conversation_messages.append(HumanMessage(content=user_input))

        # Run the multi-agent bot with full conversation history
        try:
            final_state = None
            for chunk in compiled_workflow.stream(
                {
                    "messages": conversation_messages,  # Pass full history
                    "user_profile": None  # Initialize user_profile
                },
                config={"recursion_limit": 50}
            ):
                pretty_print_messages(chunk, last_message=True)
                # Keep track of the final state
                final_state = chunk
            
            # Update conversation history with agent responses
            if final_state:
                for node_name, node_update in final_state.items():
                    if "messages" in node_update:
                        # Add the new agent messages to our history
                        new_messages = node_update["messages"]
                        conversation_messages.extend(new_messages)
        
        except Exception as e:
            print(f"❌ Error: {e}")