import arxiv, json, time
from pathlib import Path

QUERIES = [
    "classifier-free guidance diffusion",
    "latent diffusion models",
    "denoising diffusion probabilistic models",
    "DDIM sampling diffusion",
    "diffusion transformer image generation",
    "score-based generative models",
    "text-to-image diffusion models",
    "conditional diffusion image synthesis",
    "diffusion model training efficiency",
    "guidance scale sampling diffusion",
]
MAX_PER_QUERY = 40

OUT = Path("data/papers.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

client = arxiv.Client(page_size=50, delay_seconds=5.0, num_retries=5)
seen, papers = set(), []

for q in QUERIES:
    print(f"Fetching: {q}")
    search = arxiv.Search(query=q, max_results=MAX_PER_QUERY,
                          sort_by=arxiv.SortCriterion.Relevance)
    try:
        for r in client.results(search):
            pid = r.entry_id.split("/")[-1]
            if pid in seen:
                continue
            seen.add(pid)
            papers.append({
                "id": pid,
                "title": r.title.strip().replace("\n", " "),
                "abstract": r.summary.strip().replace("\n", " "),
                "authors": [a.name for a in r.authors],
                "categories": r.categories,
                "published": r.published.date().isoformat(),
                "url": r.entry_id,
            })
    except Exception as e:
        print(f"  skipped '{q}' ({e})")
    OUT.write_text(json.dumps(papers, indent=2))      # save progress each query
    print(f"  running total: {len(papers)}")
    time.sleep(3)

print(f"\nSaved {len(papers)} unique papers to {OUT}")