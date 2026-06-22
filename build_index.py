# downloads all-MiniLM-L6-v2 model

import json
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

papers = json.loads(Path("data/papers.json").read_text())
texts = [f"{p['title']}. {p['abstract']}" for p in papers]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
embeddings = np.asarray(embeddings, dtype="float32")

index = faiss.IndexFlatIP(embeddings.shape[1])   # normalized vectors + inner product = cosine
index.add(embeddings)

faiss.write_index(index, "data/papers.index")
print(f"Indexed {index.ntotal} papers, dim={embeddings.shape[1]}")