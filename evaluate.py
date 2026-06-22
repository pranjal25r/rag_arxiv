import json, re, time
import numpy as np
import faiss
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

load_dotenv()

MODEL = "llama-3.3-70b-versatile"
MIN_INTERVAL = 2.5                     # Groq free tier = 30 RPM (1 per 2s)
TOP_K = 4
FETCH_K = 20
ABSTAIN = "i don't have enough information"
print(f">>> ablation eval | provider=groq | model={MODEL} <<<\n")

papers = json.loads(Path("data/papers.json").read_text())
index = faiss.read_index("data/papers.index")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
client = Groq()

_last = [0.0]
def _throttle():
    wait = MIN_INTERVAL - (time.time() - _last[0])
    if wait > 0: time.sleep(wait)
    _last[0] = time.time()

def ask_llm(prompt, retries=6):
    for attempt in range(retries):
        _throttle()
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                w = min(60, 5 * 2 ** attempt)
                print(f"   ...{type(e).__name__}, waiting {w}s")
                time.sleep(w); continue
            raise

def retrieve_naive(question):
    q = np.asarray(embedder.encode([question], normalize_embeddings=True), dtype="float32")
    _, idx = index.search(q, TOP_K)
    return [papers[i] for i in idx[0]]

def retrieve_reranked(question):
    q = np.asarray(embedder.encode([question], normalize_embeddings=True), dtype="float32")
    _, idx = index.search(q, FETCH_K)
    cands = [papers[i] for i in idx[0]]
    pairs = [(question, f"{p['title']}. {p['abstract']}") for p in cands]
    scores = reranker.predict(pairs)
    ranked = [p for _, p in sorted(zip(scores, cands), key=lambda x: x[0], reverse=True)]
    return ranked[:TOP_K]

def generate(question, ctx):
    context = "\n\n".join(f"[{n+1}] {p['title']}\n{p['abstract']}" for n, p in enumerate(ctx))
    return ask_llm('Answer using ONLY the context. If it does not contain the answer, say exactly: '
        '"I don\'t have enough information to answer that." Cite sources by [number].\n\n'
        f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")

def judge(question, answer, ctx):
    context = "\n\n".join(f"[{i+1}] {p['title']}\n{p['abstract']}" for i, p in enumerate(ctx))
    raw = ask_llm("Evaluate this RAG answer. Return ONLY JSON with three 0.0-1.0 floats:\n"
        '"faithfulness": are all claims supported by the context?\n'
        '"answer_relevancy": does the answer address the question?\n'
        f'"context_precision": of the {len(ctx)} numbered contexts, the fraction relevant.\n\n'
        f"Contexts:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\n"
        '{"faithfulness": <float>, "answer_relevancy": <float>, "context_precision": <float>}')
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    try: return json.loads(m.group(0))
    except Exception: return {}

eval_set = json.loads(Path("eval_questions.json").read_text())

def run_eval(name, retrieve):
    print(f"\n--- {name} ---")
    faith, rel, cprec = [], [], []
    ok = tot = 0
    for item in eval_set:
        q, qtype = item["question"], item["type"]
        try:
            ctx = retrieve(q)
            answer = generate(q, ctx)
            abstained = ABSTAIN in answer.lower()
            if qtype == "out_domain":
                tot += 1; ok += int(abstained)
                print(f"[out] {'refused ✓' if abstained else 'ANSWERED ✗'}  {q}")
            elif abstained:
                print(f"[in ] abstained ✗  {q}")
            else:
                s = judge(q, answer, ctx)
                if s.get("faithfulness") is not None:
                    faith.append(s["faithfulness"]); rel.append(s["answer_relevancy"])
                    if s.get("context_precision") is not None: cprec.append(s["context_precision"])
                print(f"[in ] faith={s.get('faithfulness')} rel={s.get('answer_relevancy')} ctx_prec={s.get('context_precision')}  {q}")
        except Exception as e:
            print(f"[skip] {q} ({type(e).__name__})")
    return {"config": name, "n_scored": len(faith),
            "faithfulness": round(float(np.mean(faith)),3) if faith else None,
            "answer_relevancy": round(float(np.mean(rel)),3) if rel else None,
            "context_precision": round(float(np.mean(cprec)),3) if cprec else None,
            "abstention_accuracy": round(ok/tot,3) if tot else None}

rows = [run_eval("naive_dense", retrieve_naive),
        run_eval("reranked", retrieve_reranked)]
Path("results_ablation.json").write_text(json.dumps(rows, indent=2))

print("\n=== ABLATION ===")
hdr = f"{'config':<14}{'faith':>7}{'ans_rel':>9}{'ctx_prec':>10}{'abstain':>9}"
print(hdr); print("-"*len(hdr))
for r in rows:
    print(f"{r['config']:<14}{str(r['faithfulness']):>7}{str(r['answer_relevancy']):>9}{str(r['context_precision']):>10}{str(r['abstention_accuracy']):>9}")