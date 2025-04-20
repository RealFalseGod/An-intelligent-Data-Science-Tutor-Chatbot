from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import pipeline  # For summarization
import torch 

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
index = None
chunks = []
summarizer = None

def initialize_chatbot():
    global model, index, chunks, summarizer

    # Check if GPU is available
    print(f"Using device: {device}")

    # Load the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    model = model.to(device)

    # Load the FAISS index
    print("Loading FAISS index...")
    index = faiss.read_index("faiss_index.index")

    # Load the saved chunks
    print("Loading chunks...")
    with open("chunks.txt", "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n")  # Splitting by double newline to get each chunk

    # Load the summarization pipeline
    print("Loading summarization pipeline...")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)


    print("Chatbot initialized successfully!")



def process_input(user_input):
    try:
        # Ensure the chatbot is initialized
        if model is None or index is None or summarizer is None:
            raise ValueError("Chatbot is not initialized. Call initialize_chatbot() first.")

        # Check for empty input
        if not user_input.strip():
            return "Error: Input cannot be empty. Please provide a valid query."

        # Generate query embedding
        query_embedding = model.encode([user_input])
        query_embedding = np.array(query_embedding, dtype=np.float32)

        # Debug: Check tensor shapes
        print(f"Query embedding shape: {query_embedding.shape}")
        print(f"FAISS index dimension: {index.d}")

        # Check if FAISS index is empty
        if index.ntotal == 0:
            return "Error: The knowledge base is empty. Please upload a PDF to populate the knowledge base."

        # Search for the most relevant chunk
        D, I = index.search(query_embedding, k=1)  # k=1 means we are looking for the closest chunk

        # Handle cases where no results are found
        if I[0][0] == -1:
            return "Sorry, I couldn't find any relevant information in the knowledge base."

        # Retrieve the most similar chunk
        most_similar_index = I[0][0]
        most_similar_chunk = chunks[most_similar_index]

        # Summarize the most relevant chunk
        summary = summarizer(most_similar_chunk, max_length=100, min_length=30, do_sample=False)
        return summary[0]['summary_text']

    except ValueError as ve:
        return f"Error: {str(ve)}"

    except IndexError:
        return "Error: Unable to retrieve relevant information. The knowledge base might be empty."

    except RuntimeError as re:
        return f"Runtime Error: {str(re)}. If you're using a GPU, ensure sufficient memory is available."

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

