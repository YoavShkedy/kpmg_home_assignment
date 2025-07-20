#!/usr/bin/env python3
"""
Script to build FAISS index from HTML files in phase2_data folder
"""

import os
import pickle
import sys
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Add parent directory to path to import models
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

class IndexBuilder:
    """Class to build FAISS index from HTML files"""
    def __init__(self, data_folder: str = "phase2_data", vector_store_path: str = "part_2/indexes"):
        self.data_folder = data_folder
        self.vector_store_path = vector_store_path
        # Use Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",
            api_version="2024-12-01-preview"
        )
        # Use RecursiveCharacterTextSplitter to split text into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        
    def extract_text_from_html(self, html_file: str) -> tuple[str, Dict[str, Any]]:
        """Extract text content from HTML file
        
        Args:
            html_file: Path to the HTML file
            
        Returns:
            tuple[str, Dict[str, Any]]: Tuple containing the text content and metadata
        """
        try:
            # Read HTML file
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse HTML content with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text().strip() if title else ""
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Extract metadata
            filename = os.path.basename(html_file)
            metadata = {
                "source": filename,
                "title": title_text,
                "file_path": html_file
            }
            
            return text, metadata
            
        except Exception as e:
            print(f"Error processing {html_file}: {e}")
            return "", {}
    
    def load_documents(self) -> tuple[List[str], List[Dict[str, Any]]]:
        """Load and process all HTML documents
        
        Returns:
            tuple[List[str], List[Dict[str, Any]]]: Tuple containing the list of documents and metadata
        """
        documents = []
        metadata_list = []
        
        # Check if data folder exists
        if not os.path.exists(self.data_folder):
            print(f"Data folder {self.data_folder} not found.")
            return documents, metadata_list
        
        # Get all HTML files in data folder
        html_files = list(Path(self.data_folder).glob("*.html"))
        
        # Check if there are any HTML files
        if not html_files:
            print(f"No HTML files found in {self.data_folder}")
            return documents, metadata_list
        
        print(f"Found {len(html_files)} HTML files")
        
        # Process each HTML file
        for html_file in html_files:
            print(f"Processing {html_file}")
            # Extract text content and metadata from HTML file
            text, metadata = self.extract_text_from_html(str(html_file))
            
            # Check if text content is not empty
            if text.strip():
                # Split text into chunks using RecursiveCharacterTextSplitter
                chunks = self.text_splitter.split_text(text)
                
                # Add each chunk to the documents list and metadata list
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        documents.append(chunk)
                        chunk_metadata = metadata.copy()
                        chunk_metadata["chunk_id"] = i
                        chunk_metadata["total_chunks"] = len(chunks)
                        metadata_list.append(chunk_metadata)
            else:
                print(f"No text extracted from {html_file}")
        
        print(f"Total document chunks: {len(documents)}")
        return documents, metadata_list
    
    def build_index(self):
        """Build FAISS index from documents"""
        print("Loading documents...")
        documents, metadata_list = self.load_documents()
        
        if not documents:
            print("No documents to index.")
            return
        
        print("Generating embeddings...")
        try:
            # Generate embeddings for each document
            embeddings_list = self.embeddings.embed_documents(documents)
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return
        
        print(f"Embeddings shape: {embeddings_array.shape}")
        
        # Create FAISS index
        dimension = embeddings_array.shape[1]
        # Create FAISS index with Inner Product (cosine similarity)
        index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add embeddings to index
        index.add(embeddings_array) # type: ignore
        
        print(f"Created FAISS index with {index.ntotal} documents")
        
        # Create vector store directory
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Create paths for index, documents, and metadata
        index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
        docs_path = os.path.join(self.vector_store_path, "documents.pkl")
        metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
        
        print("Saving index and metadata...")
        
        # Save index
        faiss.write_index(index, index_path)
        
        # Save documents
        with open(docs_path, 'wb') as f:
            pickle.dump(documents, f)
        
        # Save metadata
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata_list, f)
        
        print(f"Index saved to {self.vector_store_path}")
        
        # Print statistics
        print("\nIndex Statistics:")
        print(f"Total documents: {len(documents)}")
        print(f"Index dimension: {dimension}")
        print(f"Index type: {type(index).__name__}")

def main():
    """Main function"""
    print("Building FAISS index from HTML files...")
    
    builder = IndexBuilder()
    builder.build_index()
    
    print("Done!")

if __name__ == "__main__":
    main()