import streamlit as st
import json
import io
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI
from ocr import DocumentOCRProcessor
from field_extraction import FieldExtractor

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Israeli National Insurance Form Extractor",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        margin: 1rem 0;
    }
    .json-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        font-family: 'Courier New', monospace;
    }
    .file-metrics {
        font-size: 0.8rem;
    }
    .file-metrics .metric-value {
        font-size: 1.0rem;
    }
    .file-metrics .metric-label {
        font-size: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)

class StreamlitApp:
    """
    Streamlit application for Israeli National Insurance form extraction
    """
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_azure_clients()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'extraction_result' not in st.session_state:
            st.session_state.extraction_result = None
        if 'ocr_result' not in st.session_state:
            st.session_state.ocr_result = None
        if 'validation_result' not in st.session_state:
            st.session_state.validation_result = None
    
    def setup_azure_clients(self):
        """Setup Azure clients"""
        try:
            # Get environment variables with error checking
            doc_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
            doc_api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
            
            if not doc_endpoint or not doc_api_key:
                raise ValueError("Missing required environment variables: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY")
            
            # Initialize Azure OpenAI
            self.llm = AzureChatOpenAI(
                azure_deployment="gpt-4o",
                api_version="2024-12-01-preview",
                temperature=0
            )
            
            # Initialize OCR and field extraction processors
            self.ocr_processor = DocumentOCRProcessor(
                endpoint=doc_endpoint,
                api_key=doc_api_key
            )
            self.field_extractor = FieldExtractor(self.llm)
            
            st.session_state.setup_complete = True
            
        except Exception as e:
            st.error(f"Error setting up Azure clients: {str(e)}")
            st.session_state.setup_complete = False
    
    def render_header(self):
        """Render the application header"""
        st.markdown('<h1 class="main-header">📋 Israeli National Insurance Form Extractor</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <strong>About:</strong> This application extracts information from Israeli National Insurance Institute (ביטוח לאומי) forms 
            using Azure Document Intelligence for OCR and Azure OpenAI for field extraction.
            <br><br>
            <strong>Supported formats:</strong> PDF, JPG, JPEG, PNG, TIFF
            <br>
            <strong>Languages:</strong> Hebrew and English
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with configuration options"""
        
        # Instructions
        st.sidebar.header("📝 Instructions")
        st.sidebar.markdown("""
        1. Upload a PDF or image file
        2. Click 'Process Document'
        3. Review the extracted fields
        4. Review the validation report
        5. Download the JSON result
        """)
        
        return {
            "show_ocr_text": False,
            "show_validation": True,
            "show_raw_response": False
        }
    
    def handle_file_upload(self):
        """Handle file upload and processing"""
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=['pdf', 'jpg', 'jpeg', 'png', 'tiff'],
            help="Upload a PDF or image file containing an Israeli National Insurance form"
        )
        
        if uploaded_file is not None:
            # Process button
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                return self.process_document(uploaded_file)
        
        return None
    
    def process_document(self, uploaded_file):
        """Process the uploaded document"""
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()    
            
            # Step 1: OCR Processing
            status_text.text("🔍 Extracting text with Azure Document Intelligence...")
            progress_bar.progress(25)
            
            # Read file content
            file_content = uploaded_file.read()
            
            # OCR processing
            ocr_result = self.ocr_processor.extract_text_from_document(
                file_content, uploaded_file.type
            )
            
            if not ocr_result["success"]:
                st.error(f"OCR processing failed: {ocr_result.get('error', 'Unknown error')}")
                return None
            
            st.session_state.ocr_result = ocr_result
            
            # Step 2: Field Extraction
            status_text.text("🤖 Extracting fields with Azure OpenAI...")
            progress_bar.progress(60)
            
            # Field extraction
            extraction_result = self.field_extractor.extract_fields(
                ocr_result["extracted_text"]
            )
            
            if not extraction_result["success"]:
                st.error(f"Field extraction failed: {extraction_result.get('error', 'Unknown error')}")
                return None
            
            st.session_state.extraction_result = extraction_result
            
            # Complete
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            return True
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            return None
    
    def render_results(self, config):
        """Render the processing results"""
        if st.session_state.extraction_result is None:
            return
        
        st.header("📊 Extraction Results")
        
        # Success message
        st.markdown('<div class="success-box">✅ Document processed successfully!</div>', 
                   unsafe_allow_html=True)
        
        # Processing summary
        extracted_fields = st.session_state.extraction_result["extracted_fields"]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            # Count fields extracted
            fields_extracted = sum(1 for k, v in extracted_fields.items() if v != "" and not isinstance(v, dict))
            # Count extracted nested fields
            for k, v in extracted_fields.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        if v2 != "":
                            fields_extracted += 1
            st.metric("Fields Extracted", fields_extracted)
        with col2:
            st.metric("Processing Status", "✅ Success")
        with col3:
            st.metric("Data Validated", "✅ Complete")
        
        # Extracted Fields (Main Result)
        st.subheader("📋 Extracted Fields")
        
        extracted_fields = st.session_state.extraction_result["extracted_fields"]
        
        # Display as formatted JSON
        json_str = json.dumps(extracted_fields, indent=2, ensure_ascii=False)
        st.markdown(f'<div class="json-container"><pre>{json_str}</pre></div>', 
                   unsafe_allow_html=True)
        
        # Download button
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name="extracted_fields.json",
            mime="application/json",
            use_container_width=True
        )
        
        # Validation Report - Always shown
        st.subheader("📋 Validation Report")
        validation_warnings = st.session_state.extraction_result.get("validation_warnings", [])
        
        if validation_warnings:
            st.markdown('<div class="warning-box">⚠️ Validation warnings detected. Please review the fields below:</div>', 
                       unsafe_allow_html=True)
            
            # Group warnings by field
            warning_data = []
            for warning in validation_warnings:
                warning_data.append({
                    "Field": warning["field"],
                    "Issue": warning["message"],
                    "Value": warning.get("value", "")
                })
            
            # Display warnings in a table
            if warning_data:
                import pandas as pd
                df_warnings = pd.DataFrame(warning_data)
                st.dataframe(df_warnings, use_container_width=True)
        else:
            st.markdown('<div class="success-box">✅ All fields passed validation checks</div>', 
                       unsafe_allow_html=True)
    
    def run(self):
        """Run the Streamlit application"""
        # Render header
        self.render_header()
        
        # Check setup
        if not st.session_state.get('setup_complete', False):
            st.error("❌ Azure client setup failed. Please check your configuration.")
            return
        
        # Render sidebar
        config = self.render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("📤 Upload Document")
            result = self.handle_file_upload()
        
        with col2:
            if st.session_state.extraction_result:
                self.render_results(config)
            else:
                st.header("📋 Results")
                st.info("Upload and process a document to see results here.")

def main():
    """Main function to run the Streamlit app"""
    app = StreamlitApp()
    app.run()

if __name__ == "__main__":
    main()