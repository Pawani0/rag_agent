from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import faiss
import numpy as np
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
INDEX_PATH = "data/embeddings/healthcare_index.index"
DATA_PATH = "data/embeddings/healthcare_data.pkl"


class HealthcareKnowledgeBase:
    """Healthcare knowledge base with FAISS retrieval"""
    
    def __init__(self, model_id="google/embeddinggemma-300M"):
        """Initialize the knowledge base"""
        self.model_id = model_id
        self.model = None
        self.index = None
        self.chunks = None
        self.metadata = None
        self.loaded = False
    
    def load(self):
        """Load the model and FAISS index"""
        if self.loaded:
            print("Knowledge base already loaded")
            return
        
        print("Loading healthcare knowledge base...")
        
        # Login to HuggingFace
        login(ACCESS_TOKEN)
        
        # Load embedding model
        print(f"Loading model: {self.model_id}")
        self.model = SentenceTransformer(self.model_id).to(device="cpu")
        
        # Load FAISS index
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Please run: python process_healthcare_pdfs.py"
            )
        
        self.index = faiss.read_index(INDEX_PATH)
        print(f"✓ Loaded FAISS index with {self.index.ntotal} vectors")
        
        # Load chunks and metadata
        with open(DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.metadata = data['metadata']
        print(f"✓ Loaded {len(self.chunks)} chunks")
        
        self.loaded = True
        print("✓ Knowledge base ready!")
    
    def search(self, query, k=3):
        """
        Search for similar chunks
        
        Args:
            query: Search query string
            k: Number of results to return
        
        Returns:
            List of dicts with rank, distance, text, and source
        """
        if not self.loaded:
            self.load()
        
        # Generate query embedding
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.index.search(query_vector, k)
        
        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            chunk = self.chunks[idx]
            meta = self.metadata[idx]
            
            results.append({
                'rank': i + 1,
                'distance': float(dist),
                'similarity_score': 1 / (1 + float(dist)),  # Convert distance to similarity
                'text': chunk.page_content if hasattr(chunk, 'page_content') else str(chunk),
                'source': meta.get('source', 'unknown'),
                'chunk_size': meta.get('chunk_size', 0)
            })
        
        return results
    
    def get_context(self, query, k=3, max_length=2000):
        """
        Get context for RAG by searching and combining results
        
        Args:
            query: Search query
            k: Number of chunks to retrieve
            max_length: Maximum context length in characters
        
        Returns:
            Combined context string and list of sources
        """
        results = self.search(query, k=k)
        
        context_parts = []
        sources = set()
        total_length = 0
        
        for result in results:
            text = result['text']
            source = result['source']
            
            # Add text if within length limit
            if total_length + len(text) <= max_length:
                context_parts.append(text)
                sources.add(source)
                total_length += len(text)
            else:
                # Add partial text to reach max_length
                remaining = max_length - total_length
                if remaining > 100:  # Only add if meaningful amount remains
                    context_parts.append(text[:remaining] + "...")
                    sources.add(source)
                break
        
        context = "\n\n".join(context_parts)
        return context, list(sources), results


# Global instance (singleton pattern)
_kb_instance = None

def get_knowledge_base():
    """Get or create the global knowledge base instance"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = HealthcareKnowledgeBase()
        _kb_instance.load()
    return _kb_instance


if __name__ == "__main__":
    # Test the knowledge base
    print("Testing Healthcare Knowledge Base...")
    
    kb = HealthcareKnowledgeBase()
    kb.load()
    
    # Test queries
    test_queries = [
        "What is diabetes?",
        "How much sleep do I need?",
        "Side effects of aspirin",
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        context, sources, results = kb.get_context(query, k=2)
        
        print(f"\nSources: {', '.join(sources)}")
        print(f"\nContext ({len(context)} chars):")
        print(context[:300] + "..." if len(context) > 300 else context)
