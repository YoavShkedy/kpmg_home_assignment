import os
import pickle
import faiss
import numpy as np
from typing import List, Dict, Any
from langchain_openai import AzureOpenAIEmbeddings
from models.schemas import RetrievalResult
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class VectorService:
    """Service to handle vector store operations"""
    def __init__(self, vector_store_path: str = "part_2/indexes"):
        self.vector_store_path = vector_store_path
        # Use Azure OpenAI embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment="text-embedding-3-small",
            api_version="2024-12-01-preview"
        )
        # Load FAISS index and associated data
        self.index = None
        self.documents = None
        self.metadata = None
        self.load_index()
    
    def load_index(self):
        """Load FAISS index and associated data"""
        try:
            # Paths for index, documents, and metadata
            index_path = os.path.join(self.vector_store_path, "faiss_index.bin")
            docs_path = os.path.join(self.vector_store_path, "documents.pkl")
            metadata_path = os.path.join(self.vector_store_path, "metadata.pkl")
            
            # Check if all files exist
            if all(os.path.exists(p) for p in [index_path, docs_path, metadata_path]):
                # Load FAISS index
                self.index = faiss.read_index(index_path)

                # Load documents
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
                    
                print(f"Loaded FAISS index with {self.index.ntotal} documents")
            else:
                print("Vector store not found. Please run build_index.py first.")
                self.index = None
                self.documents = []
                self.metadata = []
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.index = None
            self.documents = []
            self.metadata = []
    
    def search(self, query: str, k: int = 5, hmo_filter: str = "") -> List[RetrievalResult]:
        """Perform similarity search on the FAISS index
        
        Args:
            query: The query to search with
            k: The number of results to return
            hmo_filter: Optional HMO name to filter results by
        
        Returns:
            List[RetrievalResult]: List of retrieval results
        """
        # Check if index is loaded
        if self.index is None:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            # Convert query embedding to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search in FAISS
            scores, indexes = self.index.search(query_vector, min(k, self.index.ntotal))
            
            results = []
            # Iterate over retrieved scores and indexes
            for score, idx in zip(scores[0], indexes[0]):
                
                # Get document and metadata
                doc = self.documents[idx] # type: ignore
                metadata = self.metadata[idx] # type: ignore
                
                # Apply HMO filter if provided
                if hmo_filter:
                    # Check if the document's HMO matches the filter
                    doc_hmo = metadata.get('hmo', '').lower()
                    filter_hmo = hmo_filter.lower()
                    
                    # Skip if HMO doesn't match (allow partial matches)
                    if filter_hmo not in doc_hmo and doc_hmo not in filter_hmo:
                        continue
                
                # Add retrieval result
                results.append(RetrievalResult(
                    content=doc,
                    metadata=metadata,
                    score=float(score) # Cosine similarity score
                ))
                
                if len(results) >= k:
                    break
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics
        
        Returns:
            Dict[str, Any]: Dictionary containing vector store statistics
        """
        # Check if index is loaded
        if self.index is None:
            return {"status": "not_loaded", "total_documents": 0}
        
        return {
            "status": "loaded",
            "total_documents": self.index.ntotal,
            "dimension": self.index.d,
            "index_type": type(self.index).__name__
        }