#This can be used on collab due to local system limitations
#!pip install -q faiss-cpu sentence-transformers PyPDF2

from sentence_transformers import SentenceTransformer
from functions_for_data import *
import os
import faiss
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu' 

index_file = "faiss_index.index"
if os.path.exists(index_file):
    index = faiss.read_index(index_file)
else:
    
    index = faiss.IndexFlatL2(384)  

  
    if device == 'cuda':
        res = faiss.StandardGpuResources()  
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  
    else:
        gpu_index = index

chunks = []
if os.path.exists("chunks.txt"):
    with open("chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n")

def extract_text_from_pdf(pdf_path):
   
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device) 
    
    
    text = extract_text_from_pdf(pdf_path)
    print("Extracted text from PDF") 
    new_chunks = create_chunks(text)
    
    
    new_embeddings = create_embeddings(new_chunks, model) 
    
    
    if device == 'cuda':
        new_embeddings = torch.tensor(new_embeddings).cuda()  
        new_embeddings = new_embeddings.cpu().numpy()  
    else:
        new_embeddings = torch.tensor(new_embeddings).numpy()

  
    gpu_index.add(new_embeddings) 

    global chunks
    chunks.extend(new_chunks)
    
  
    with open("chunks.txt", "a", encoding="utf-8") as f:
        for chunk in new_chunks:
            f.write(f"{chunk}\n\n")
    
    if device == 'cuda':
        faiss.write_index(gpu_index, "faiss_index.index") 
    else:
        faiss.write_index(gpu_index, "faiss_index.index")  
    print("Data uploaded to index")
