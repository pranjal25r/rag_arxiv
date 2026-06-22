import json
import sys
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

load_dotenv()

papers = json.loads(Path("data/papers.json").read_text())
index = faiss.read_index("data/papers.index")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = genai.Client()

TOP_K = 4

def answer(question: str):
    q_emb = np.asarray(model.encode([question], normalize_embeddings=True), dtype="float32")
    scores, idx = index.search(q_emb, TOP_K)
    retrieved = [papers[i] for i in idx[0]]

    context = "\n\n".join(
        f"[{n+1}] {p['title']}\n{p['abstract']}" for n, p in enumerate(retrieved)
    )
    prompt = f"""Answer the question using ONLY the context below.
If the context does not contain the answer, say exactly: "I don't have enough information to answer that."
Cite the sources you use by their [number].

Context:
{context}

Question: {question}

Answer:"""

    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    print(response.text)
    print("\n--- Retrieved ---")
    for n, (p, s) in enumerate(zip(retrieved, scores[0])):
        print(f"[{n+1}] ({s:.3f}) {p['title']}")

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What problem do diffusion models solve?"
    answer(q)