import os
import json 
import logging
import io

from typing import Dict, Any
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient

class DocumentOCRProcessor:
    """
    Azure Document Intelligence OCR processor for extracting text from documents
    """
    
    def __init__(self, endpoint: str, api_key: str):
        """
        Initialize the Document Intelligence client
        
        Args:
            endpoint: Azure Document Intelligence endpoint
            api_key: Azure Document Intelligence API key
        """
        self.client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(api_key)
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        # Disable Azure SDK logging
        logging.getLogger('azure').setLevel(logging.WARNING)
        self.logger = logging.getLogger(__name__)
    
    def extract_text_from_document(self, file_content: bytes, content_type: str) -> Dict[str, Any]:
        """
        Extract text from document using Azure Document Intelligence
        
        Args:
            file_content: Binary content of the file
            content_type: MIME type of the file (e.g., 'application/pdf', 'image/jpeg')
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            self.logger.info(f"Starting OCR extraction for document type: {content_type}")
            
            # Use the layout model for comprehensive text extraction
            # Convert bytes to BytesIO for Azure SDK
            file_stream = io.BytesIO(file_content)
            poller = self.client.begin_analyze_document(
                model_id="prebuilt-layout",
                body=file_stream,
                content_type=content_type
            )
            
            result = poller.result()
            
            # Extract all text content
            extracted_text = ""
            if result.content:
                extracted_text = result.content
            
            # Extract structured information
            structured_data = {
                "content": extracted_text,
                "pages": [],
                "tables": [],
                "key_value_pairs": []
            }
            
            # Process pages
            if result.pages:
                for page in result.pages:
                    page_info = {
                        "page_number": page.page_number,
                        "text": "",
                        "lines": []
                    }
                    
                    if page.lines:
                        for line in page.lines:
                            page_info["lines"].append({
                                "text": line.content,
                                "bounding_box": line.polygon if hasattr(line, 'polygon') else None
                            })
                            page_info["text"] += line.content + "\n"
                    
                    structured_data["pages"].append(page_info)
            
            # Process tables if any
            if result.tables:
                for table in result.tables:
                    table_data = {
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "cells": []
                    }
                    
                    if table.cells:
                        for cell in table.cells:
                            table_data["cells"].append({
                                "content": cell.content,
                                "row_index": cell.row_index,
                                "column_index": cell.column_index
                            })
                    
                    structured_data["tables"].append(table_data)
            
            # Process key-value pairs if any
            if result.key_value_pairs:
                for kv_pair in result.key_value_pairs:
                    kv_data = {
                        "key": kv_pair.key.content if kv_pair.key else "",
                        "value": kv_pair.value.content if kv_pair.value else ""
                    }
                    structured_data["key_value_pairs"].append(kv_data)
            
            self.logger.info(f"OCR extraction completed. Extracted {len(extracted_text)} characters")
            
            return {
                "success": True,
                "extracted_text": extracted_text,
                "structured_data": structured_data,
                "page_count": len(result.pages) if result.pages else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error during OCR extraction: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "structured_data": None
            }
    
    def extract_from_file_path(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a file given its path
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            # Determine content type based on file extension
            _, ext = os.path.splitext(file_path.lower())
            content_type_map = {
                '.pdf': 'application/pdf',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.tiff': 'image/tiff',
                '.tif': 'image/tiff'
            }
            
            content_type = content_type_map.get(ext)
            if not content_type:
                raise ValueError(f"Unsupported file type: {ext}")
            
            # Read file content
            with open(file_path, 'rb') as file:
                file_content = file.read()
            
            return self.extract_text_from_document(file_content, content_type)
            
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "extracted_text": "",
                "structured_data": None
            }


if __name__ == "__main__":
    # load environment variables from .env file
    load_dotenv()

    # Get environment variables with error handling
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    api_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    if not endpoint or not api_key:
        raise ValueError("Missing required environment variables: AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT, AZURE_DOCUMENT_INTELLIGENCE_KEY")

    # Create a client for the Document Intelligence service
    ocr = DocumentOCRProcessor(
        endpoint=endpoint,
        api_key=api_key
    )

    # Extract text from the file
    result = ocr.extract_from_file_path("phase1_data/283_ex1.pdf")

    # Output result to a text file
    with open("part_1/ocr_test_result.txt", "w", encoding="utf-8") as f:
        f.write(result["extracted_text"])