import os
import json
import faiss
import numpy as np

def search_career_match(query_embedding, k=1):
    # Correct path to FAISS index (inside roadmap_module)
    index_path = os.path.join(os.path.dirname(__file__), "roadmap_index_local.faiss")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index file not found at: {index_path}")

    index = faiss.read_index(index_path)

    # Load career roadmap data
    roadmap_json_path = os.path.join(os.path.dirname(__file__), "career_roadmaps_full.json")
    with open(roadmap_json_path, "r") as f:
        roadmap_data = json.load(f)

    # Ensure query embedding is a 2D float32 array
    if isinstance(query_embedding, list):
        query_embedding = np.array(query_embedding)
    if query_embedding.ndim == 1:
        query_embedding = np.expand_dims(query_embedding, axis=0)
    query_embedding = query_embedding.astype("float32")

    D, I = index.search(query_embedding, k=k)

    top_index = I[0][0]
    top_score = D[0][0]

    keys = list(roadmap_data.keys())
    best_role = keys[top_index]
    role_data = roadmap_data[best_role]

    # ✅ Add role and score into returned dict
    return {
        "role": best_role,
        "score": float(top_score),
        "description": role_data.get("description", ""),
        "roadmap": role_data.get("roadmap", [])
    }
