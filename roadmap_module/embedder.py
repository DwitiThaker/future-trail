import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_resume_embedding(text):
    return model.encode([text], convert_to_numpy=True).astype("float32")