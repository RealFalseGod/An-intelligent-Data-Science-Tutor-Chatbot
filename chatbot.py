from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline  # For summarization
import torch 

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to(device)

# Load the saved FAISS index
index = faiss.read_index("faiss_index.index")

# Load the saved chunks
chunks = []
with open("chunks.txt", "r", encoding="utf-8") as f:
    chunks = f.read().split("\n\n")  # Splitting by double newline to get each chunk

# Load a summarization pipeline and move it to GPU
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if device == "cuda" else -1)

# Function to process the user input and find the most relevant chunk
def process_input(user_input):
   
    query_embedding = model.encode([user_input], device=device)

    
    query_embedding = np.array(query_embedding, dtype=np.float32)

   
    D, I = index.search(query_embedding, k=1)  # k=1 means we are looking for the closest chunk

    
    most_similar_index = I[0][0]

    
    most_similar_chunk = chunks[most_similar_index]

   
    summary = summarizer(most_similar_chunk, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

user_input = input("Ask a question: ")
response = process_input(user_input)
print(f"Chatbot: {response}")