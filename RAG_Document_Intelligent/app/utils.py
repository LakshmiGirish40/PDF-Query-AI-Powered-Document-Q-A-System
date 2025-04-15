import fitz  # PyMuPDF
import numpy as np
import faiss
import requests
import os
from sentence_transformers import SentenceTransformer

# Load the embedding model only once (this improves performance)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def embed_text(text):
    """Embed a chunk of text using the SentenceTransformer."""
    return embedding_model.encode(text)

def create_vector_database(text_chunks):
    """Create a FAISS index and add embeddings."""
    embeddings = np.array([embed_text(chunk) for chunk in text_chunks])
    index = faiss.IndexFlatL2(embeddings.shape[1])  # Using L2 distance metric for FAISS
    index.add(embeddings)
    return index, embeddings

def retrieve_relevant_chunks(question, index, text_chunks, k=3):
    """Retrieve the top k relevant text chunks for a given question."""
    question_embedding = embed_text(question)
    D, I = index.search(np.array([question_embedding]), k)  # Search for top k relevant chunks
    return [text_chunks[i] for i in I[0]]

def query_groq_llm(question, context, model="llama3-8b-8192", api_key=None):
    """Query the Groq LLM API with the context and question."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 100
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

