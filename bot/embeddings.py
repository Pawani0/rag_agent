from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle

load_dotenv()

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def get_embedding(text, model, task_type="retrieval_document"):
    embedding = model.encode(text, convert_to_tensor=False)
    return embedding

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([text])
    return chunks

def pdf_to_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def txt_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def read_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return pdf_to_text(file_path)
    elif file_extension == '.txt':
        return txt_to_text(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Only PDF and TXT are supported.")

def create_faiss_index(embeddings):
    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    print(f"FAISS index created with {index.ntotal} vectors of dimension {dimension}")
    return index

def save_index(index, chunks, embeddings, index_path="faiss_index.index", data_path="chunks_data.pkl"):
    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")
    
    # Save chunks and embeddings
    with open(data_path, 'wb') as f:
        pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)
    print(f"Chunks and embeddings saved to {data_path}")

def load_index(index_path="faiss_index.index", data_path="chunks_data.pkl"):
    # Load FAISS index
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from {index_path}")
    
    # Load chunks and embeddings
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Chunks and embeddings loaded from {data_path}")
    
    return index, data['chunks'], data['embeddings']

def search_similar(query, model, index, chunks, k=3):
    # Generate embedding for query
    query_embedding = get_embedding(query, model)
    query_vector = np.array([query_embedding]).astype('float32')
    
    # Search FAISS index
    distances, indices = index.search(query_vector, k)
    
    # Return results
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        results.append({
            'rank': i + 1,
            'distance': float(dist),
            'text': chunks[idx].page_content if hasattr(chunks[idx], 'page_content') else str(chunks[idx])
        })
    
    return results

def main():
    # Login to HuggingFace
    login(ACCESS_TOKEN)
    
    # Setup device and model
    device = "cpu"
    
    model_id = "google/embeddinggemma-300M"
    print(f"Loading model: {model_id}")
    model = SentenceTransformer(model_id).to(device=device)
    
    # Read file (PDF or TXT)
    file_path = "askgalore_cleaned.txt"
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    print(f"\nReading file: {file_path}")
    text = read_file(file_path)
    print(f"Extracted {len(text)} characters")
    
    # Chunk the text
    print(f"\nChunking text with size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    chunks = chunk_text(text)
    print(f"Created {len(chunks)} chunks")
    
    # Generate embeddings for all chunks
    print(f"\nGenerating embeddings for {len(chunks)} chunks...")
    embeddings = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk.page_content, model)
        embeddings.append(embedding)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(chunks)} chunks")
    
    print(f"Generated {len(embeddings)} embeddings")
    
    # Create FAISS index
    print(f"\nCreating FAISS index...")
    index = create_faiss_index(embeddings)
    
    # Save index and data
    print(f"\nSaving index and data...")
    save_index(index, chunks, embeddings, "askgalore_index.index", "askgalore_data.pkl")
    
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Files created:")
    print(f"  - askgalore_index.index (FAISS index)")
    print(f"  - askgalore_data.pkl (chunks and embeddings)")

if __name__ == "__main__":
    main()
