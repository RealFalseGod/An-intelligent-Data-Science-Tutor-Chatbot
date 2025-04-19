from sentence_transformers import SentenceTransformer
from functions_for_data import *
# Load the pre-trained model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example of extracting text and creating chunks and embeddings
pdf_path = "data_science_notes.pdf"
text = extract_text_from_pdf(pdf_path)

chunks = create_chunks(text)
embeddings = create_embeddings(chunks,model)

# Build the FAISS index for efficient similarity search
index = build_index(embeddings)

# Save the index for future use
faiss.write_index(index, "faiss_index.index")

# Save the chunks as well, so we can reference them later
with open("chunks.txt", "w",encoding="utf-8") as f:
    for chunk in chunks:
        f.write(f"{chunk}\n\n")
