import faiss
import numpy as np
import fitz

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def create_chunks(text, chunk_size=2000):  # Increase chunk size to 2000
    """Split the text into chunks of specified size using a direct for loop"""
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def create_embeddings(chunks,model):
    """Convert text chunks into vector embeddings"""
    embeddings = model.encode(chunks, convert_to_numpy=True)
    return embeddings

def build_index(embeddings):
    """Build the FAISS index for similarity search"""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index