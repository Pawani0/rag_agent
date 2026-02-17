from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
import faiss
import numpy as np
import pickle
from pathlib import Path
import fitz  # PyMuPDF
import easyocr
from PIL import Image
import io

load_dotenv()

# Configuration
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")  # HuggingFace token
PDF_DIR = "data/documents/pdfs_essential"
OUTPUT_DIR = "data/embeddings"
INDEX_PATH = os.path.join(OUTPUT_DIR, "healthcare_index.index")
DATA_PATH = os.path.join(OUTPUT_DIR, "healthcare_data.pkl")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# OCR threshold: if text extraction yields fewer characters per page, use OCR
OCR_THRESHOLD = 50  # characters per page

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Initialize EasyOCR reader (will be loaded on demand)
ocr_reader = None


def get_ocr_reader():
    """Initialize OCR reader lazily"""
    global ocr_reader
    if ocr_reader is None:
        print("   📷 Initializing OCR reader (this may take a moment)...")
        ocr_reader = easyocr.Reader(['en'], gpu=False)  # English language, CPU mode
        print("   ✓ OCR reader initialized")
    return ocr_reader


def pdf_to_text_standard(file_path):
    """Extract text from PDF using standard method (PyPDF)"""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def pdf_to_text_ocr(file_path):
    """Extract text from PDF using OCR (for scanned/image PDFs)"""
    reader = get_ocr_reader()
    doc = fitz.open(str(file_path))
    text = ""
    
    total_pages = len(doc)
    print(f"   📄 Processing {total_pages} pages with OCR...")
    
    for page_num, page in enumerate(doc):
        # Convert page to image
        pix = page.get_pixmap(dpi=150)  # 150 DPI for good quality
        img_data = pix.tobytes("png")
        
        # Perform OCR
        results = reader.readtext(img_data, detail=0, paragraph=True)
        page_text = "\n".join(results)
        text += page_text + "\n"
        
        # Progress indicator
        if (page_num + 1) % 5 == 0 or (page_num + 1) == total_pages:
            print(f"      Processed {page_num + 1}/{total_pages} pages")
    
    doc.close()
    return text


def pdf_to_text(file_path):
    """
    Smart PDF text extraction:
    1. Try standard extraction first
    2. If minimal text found, use OCR
    """
    # Try standard extraction
    text = pdf_to_text_standard(file_path)
    
    # Check if we got meaningful text
    if len(text.strip()) < OCR_THRESHOLD:
        print(f"   ⚠️  Low text content ({len(text)} chars) - switching to OCR")
        text = pdf_to_text_ocr(file_path)
        print(f"   ✓ OCR extraction complete")
    
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.create_documents([text])
    return chunks


def get_embedding(text, model):
    """Generate embedding for text"""
    embedding = model.encode(text, convert_to_tensor=False)
    return embedding


def create_faiss_index(embeddings):
    """Create FAISS index from embeddings"""
    embeddings_array = np.array(embeddings).astype('float32')
    dimension = embeddings_array.shape[1]
    
    # Create FAISS index (L2 distance)
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    print(f"✓ FAISS index created with {index.ntotal} vectors of dimension {dimension}")
    return index


def save_index(index, chunks, embeddings, metadata, index_path=INDEX_PATH, data_path=DATA_PATH):
    """Save FAISS index and associated data"""
    # Save FAISS index
    faiss.write_index(index, index_path)
    print(f"✓ FAISS index saved to {index_path}")
    
    # Save chunks, embeddings, and metadata
    with open(data_path, 'wb') as f:
        pickle.dump({
            'chunks': chunks,
            'embeddings': embeddings,
            'metadata': metadata
        }, f)
    print(f"✓ Chunks and metadata saved to {data_path}")


def process_healthcare_pdfs():
    """Main processing function"""
    print("=" * 80)
    print("HEALTHCARE PDF PROCESSING WITH OCR SUPPORT")
    print("=" * 80)
    
    # Login to HuggingFace
    print("\n1. Logging in to HuggingFace...")
    login(ACCESS_TOKEN)
    print("✓ Logged in successfully")
    
    # Load embedding model
    print("\n2. Loading embedding model...")
    model_id = "google/embeddinggemma-300M"
    device = "cpu"
    model = SentenceTransformer(model_id).to(device=device)
    print(f"✓ Loaded model: {model_id}")
    
    # Find all PDFs
    print(f"\n3. Finding PDFs in {PDF_DIR}...")
    pdf_files = list(Path(PDF_DIR).glob("*.pdf"))
    
    if not pdf_files:
        print(f"✗ No PDF files found in {PDF_DIR}")
        print("Please download PDFs first using: python download_essential_pdfs.py")
        return
    
    print(f"✓ Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        size_mb = pdf.stat().st_size / (1024 * 1024)
        print(f"  - {pdf.name} ({size_mb:.2f} MB)")
    
    # Process each PDF
    all_chunks = []
    all_metadata = []
    
    print(f"\n4. Extracting text from PDFs (with OCR fallback)...")
    for pdf_file in pdf_files:
        print(f"\n   Processing: {pdf_file.name}")
        
        # Extract text (with smart OCR fallback)
        text = pdf_to_text(str(pdf_file))
        print(f"   ✓ Extracted {len(text):,} characters")
        
        if len(text.strip()) < 100:
            print(f"   ⚠️  WARNING: Very little text extracted - PDF may be blank or corrupted")
            continue
        
        # Chunk text
        chunks = chunk_text(text)
        print(f"   ✓ Created {len(chunks)} chunks")
        
        # Store chunks with metadata
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append({
                'source': pdf_file.name,
                'chunk_size': len(chunk.page_content)
            })
    
    if len(all_chunks) == 0:
        print("\n✗ No chunks created - no text could be extracted from PDFs")
        return
    
    print(f"\n✓ Total chunks created: {len(all_chunks)}")
    
    # Generate embeddings
    print(f"\n5. Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = []
    
    for i, chunk in enumerate(all_chunks):
        embedding = get_embedding(chunk.page_content, model)
        embeddings.append(embedding)
        
        if (i + 1) % 10 == 0 or (i + 1) == len(all_chunks):
            print(f"   Processed {i + 1}/{len(all_chunks)} chunks")
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    
    # Create FAISS index
    print(f"\n6. Creating FAISS index...")
    index = create_faiss_index(embeddings)
    
    # Save everything
    print(f"\n7. Saving index and data...")
    save_index(index, all_chunks, embeddings, all_metadata)
    
    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print("=" * 80)
    print(f"\n✓ Processed {len(pdf_files)} PDF files")
    print(f"✓ Created {len(all_chunks)} text chunks")
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"✓ FAISS index dimension: {embeddings[0].shape[0]}")
    
    print(f"\n📁 Output files:")
    print(f"  - {INDEX_PATH}")
    print(f"  - {DATA_PATH}")
    
    print(f"\n✨ Ready for retrieval!")
    print("You can now use the retrieval node in your RAG agent.")


if __name__ == "__main__":
    process_healthcare_pdfs()
