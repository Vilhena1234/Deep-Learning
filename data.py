import argparse
import csv
import json
import random
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
SQUAD_URLS = {
    "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json",
    "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _ssl_context():
    """
    SSL context that works on Windows/macOS where Python often can't find
    root certificates. Tries, in order:
      1. truststore -> uses the operating system's certificate store (best on Windows)
      2. certifi    -> Mozilla's certificate bundle
      3. Python's default
    Install one with:  pip install truststore   (or)   pip install certifi
    """
    import ssl
    try:
        import truststore
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _download(url: str, dest: Path) -> Path:
    """Download url to dest, skipping if the file already exists."""
    if dest.exists():
        print(f"  already downloaded: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, context=_ssl_context()) as r, open(tmp, "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"\nDownload failed: {e}\n"
            f"Fix: run `pip install truststore` and try again.\n"
            f"Or download it in your browser:\n  {url}\n"
            f"and save it as:\n  {dest.resolve()}\n"
            f"then run this script again (it skips files that already exist)."
        )
    tmp.rename(dest)
    print(f"  saved to {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def _save(name: str, split: str, corpus, queries, qrels) -> None:
    path = OUT_DIR / f"{name}_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"corpus": corpus, "queries": queries, "qrels": qrels}, f, ensure_ascii=False)
    print(f"  wrote {path}: {len(corpus)} docs, {len(queries)} queries")


# --------------------------------------------------------------------------- #
# FiQA (BEIR format: corpus.jsonl, queries.jsonl, qrels/{train,dev,test}.tsv)
# --------------------------------------------------------------------------- #
def prepare_fiqa() -> None:
    print("FiQA")
    zip_path = _download(FIQA_URL, RAW_DIR / "fiqa.zip")
    folder = RAW_DIR / "fiqa"
    if not folder.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW_DIR)

    # Full corpus is shared by all splits (as in BEIR)
    corpus = {}
    with open(folder / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            text = (d.get("title", "") + " " + d["text"]).strip()
            corpus[str(d["_id"])] = text

    all_queries = {}
    with open(folder / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            all_queries[str(q["_id"])] = q["text"]

    for split in ["train", "dev", "test"]:
        qrels_path = folder / "qrels" / f"{split}.tsv"
        if not qrels_path.exists():
            continue
        qrels = {}
        with open(qrels_path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # header: query-id, corpus-id, score
            for qid, did, score in reader:
                if int(score) > 0:
                    qrels.setdefault(str(qid), {})[str(did)] = int(score)
        queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
        _save("fiqa", split, corpus, queries, qrels)


# --------------------------------------------------------------------------- #
# SQuAD v1.1: each question -> its paragraph (context) is the relevant passage
# --------------------------------------------------------------------------- #
def prepare_squad() -> None:
    print("SQuAD")
    for split, url in SQUAD_URLS.items():
        path = _download(url, RAW_DIR / f"squad_{split}.json")
        with open(path, encoding="utf-8") as f:
            articles = json.load(f)["data"]

        corpus, queries, qrels = {}, {}, {}
        context_to_id = {}  # de-duplicate identical paragraphs
        for article in articles:
            for para in article["paragraphs"]:
                ctx = para["context"].strip()
                if ctx not in context_to_id:
                    doc_id = f"{split}_d{len(context_to_id)}"
                    context_to_id[ctx] = doc_id
                    corpus[doc_id] = ctx
                doc_id = context_to_id[ctx]
                for qa in para["qas"]:
                    qid = str(qa["id"])
                    queries[qid] = qa["question"].strip()
                    qrels[qid] = {doc_id: 1}
        _save("squad", split, corpus, queries, qrels)


# --------------------------------------------------------------------------- #
# Alternative: download through the HuggingFace `datasets` library
#   pip install datasets
#   python data.py --source hf
# --------------------------------------------------------------------------- #
_HF_SPLIT_NAMES = {"validation": "dev", "dev": "dev", "train": "train", "test": "test"}


def prepare_fiqa_hf() -> None:
    from datasets import load_dataset
    print("FiQA (HuggingFace)")

    corpus_ds = load_dataset("BeIR/fiqa", "corpus")["corpus"]
    corpus = {str(d["_id"]): ((d.get("title") or "") + " " + d["text"]).strip() for d in corpus_ds}

    queries_ds = load_dataset("BeIR/fiqa", "queries")["queries"]
    all_queries = {str(q["_id"]): q["text"] for q in queries_ds}

    qrels_dd = load_dataset("BeIR/fiqa-qrels")  # relevance labels live in a separate repo
    for hf_split, rows in qrels_dd.items():
        split = _HF_SPLIT_NAMES.get(hf_split, hf_split)
        qrels = {}
        for r in rows:
            if int(r["score"]) > 0:
                qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
        queries = {qid: all_queries[qid] for qid in qrels if qid in all_queries}
        _save("fiqa", split, corpus, queries, qrels)


def prepare_squad_hf() -> None:
    from datasets import load_dataset
    print("SQuAD (HuggingFace)")
    dd = load_dataset("rajpurkar/squad")  # splits: train, validation
    for hf_split, rows in dd.items():
        split = _HF_SPLIT_NAMES.get(hf_split, hf_split)
        corpus, queries, qrels, context_to_id = {}, {}, {}, {}
        for r in rows:
            ctx = r["context"].strip()
            if ctx not in context_to_id:
                doc_id = f"{split}_d{len(context_to_id)}"
                context_to_id[ctx] = doc_id
                corpus[doc_id] = ctx
            qid = str(r["id"])
            queries[qid] = r["question"].strip()
            qrels[qid] = {context_to_id[ctx]: 1}
        _save("squad", split, corpus, queries, qrels)


# --------------------------------------------------------------------------- #
# Loading for the rest of the group
# --------------------------------------------------------------------------- #
def load(name: str, split: str, subset: int | None = None, n_extra_docs: int = 500, seed: int = 0):
    """
    Load a prepared dataset.

    name:   "fiqa" or "squad"
    split:  "train", "dev" or "test" (SQuAD has only "train" and "dev")
    subset: if given, keep only this many queries, their relevant docs,
            plus n_extra_docs random other docs (for fast development).
    """
    path = OUT_DIR / f"{name}_{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python data.py` first.")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    corpus, queries, qrels = d["corpus"], d["queries"], d["qrels"]

    if subset is not None:
        rng = random.Random(seed)
        qids = rng.sample(sorted(queries), min(subset, len(queries)))
        queries = {q: queries[q] for q in qids}
        qrels = {q: qrels[q] for q in qids}
        keep = {doc for q in qids for doc in qrels[q]}
        others = [doc for doc in corpus if doc not in keep]
        keep |= set(rng.sample(others, min(n_extra_docs, len(others))))
        corpus = {doc: corpus[doc] for doc in keep}

    return corpus, queries, qrels


def stats(name: str, split: str) -> None:
    corpus, queries, qrels = load(name, split)
    doc_len = sum(len(t.split()) for t in corpus.values()) / len(corpus)
    q_len = sum(len(t.split()) for t in queries.values()) / len(queries)
    rel = sum(len(v) for v in qrels.values()) / len(qrels)
    print(f"{name:6s} {split:5s} | docs {len(corpus):7d} | queries {len(queries):6d} | "
          f"avg doc len {doc_len:6.1f} | avg query len {q_len:5.1f} | relevant/query {rel:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="print dataset statistics")
    parser.add_argument("--source", choices=["direct", "hf"], default="direct",
                        help="direct = official download links; hf = HuggingFace `datasets` library")
    args = parser.parse_args()

    if args.source == "hf":
        prepare_fiqa_hf()
        prepare_squad_hf()
    else:
        prepare_fiqa()
        prepare_squad()

    if args.stats:
        print()
        for name, split in [("fiqa", "train"), ("fiqa", "dev"), ("fiqa", "test"),
                            ("squad", "train"), ("squad", "dev")]:
            if (OUT_DIR / f"{name}_{split}.json").exists():
                stats(name, split)