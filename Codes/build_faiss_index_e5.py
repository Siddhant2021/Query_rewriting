import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import os


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--passages", required=True, help="JSONL passage file")
    parser.add_argument("--output", required=True,
                        help="FAISS index output path")
    parser.add_argument("--mapping", default=None,
                        help="Optional id mapping output")
    args = parser.parse_args()

    print("🔹 Loading E5 model...")
    model = SentenceTransformer("intfloat/e5-base-v2")

    print("🔹 Reading passages (JSONL)...")
    passages = list(read_jsonl(args.passages))

    print(f"🔹 Loaded {len(passages)} passages")

    # Keep stable integer IDs
    texts = []
    id_map = []

    for p in passages:
        texts.append("passage: " + p["text"])
        id_map.append(p["_id"])

    print("🔹 Encoding passages...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    faiss.write_index(index, args.output)

    print(f"✅ FAISS index written to {args.output}")
    print(f"📐 Index size: {index.ntotal}, dim={dim}")

    # Save ID mapping (VERY IMPORTANT)
    if args.mapping:
        with open(args.mapping, "w", encoding="utf-8") as f:
            json.dump(id_map, f)
        print(f"🧭 Passage ID mapping written to {args.mapping}")


if __name__ == "__main__":
    main()
