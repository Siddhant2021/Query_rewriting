import json
import argparse
import os
import torch
import faiss
from transformers import AutoTokenizer, AutoModel, pipeline

# ------------------------
# IO helpers
# ------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ------------------------
# LINQ-style dense retriever
# ------------------------
class LINQRetriever:
    def __init__(self, encoder_name, index_path, passages_path, device="cpu"):
        self.device = device

        # Encoder
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name).to(device)
        self.encoder.eval()

        # FAISS index
        self.index = faiss.read_index(index_path)

        # Passages
        with open(passages_path, "r", encoding="utf-8") as f:
            self.passages = json.load(f)

    def encode(self, text):
        with torch.no_grad():
            # E5 requires prefix
            text = "query: " + text

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256
            ).to(self.device)

            outputs = self.encoder(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            return emb.cpu().numpy()


    def retrieve(self, query, top_k=5):
        q_emb = self.encode(query)
        scores, ids = self.index.search(q_emb, top_k)

        results = []
        for idx in ids[0]:
            p = self.passages[str(idx)]
            results.append({
                "title": p.get("title", ""),
                "text": p.get("text", "")
            })
        return results

# ------------------------
# Answer generator
# ------------------------
def build_prompt(question, contexts):
    context_text = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(contexts)
    )

    return f"""
Answer the question using ONLY the information below.

Context:
{context_text}

Question:
{question}

Answer:
"""

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run RAG using rewritten queries + LINQ retriever"
    )
    parser.add_argument("--input", required=True, help="rewritten queries JSONL")
    parser.add_argument("--output", required=True, help="eval-ready output JSONL")
    parser.add_argument("--encoder", required=True, help="LINQ encoder checkpoint")
    parser.add_argument("--index", required=True, help="FAISS index path")
    parser.add_argument("--passages", required=True, help="passages JSON")
    parser.add_argument("--domain", required=True, help="domain name (fiqa, gov, etc.)")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    retriever = LINQRetriever(
        encoder_name=args.encoder,
        index_path=args.index,
        passages_path=args.passages,
        device=device
    )

    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        device=0 if device == "cuda" else -1,
        max_new_tokens=256
    )

    outputs = []

    for ex in read_jsonl(args.input):
        question = ex["rewritten_query"]

        contexts = retriever.retrieve(question, top_k=args.topk)

        prompt = build_prompt(question, contexts)
        answer = generator(prompt)[0]["generated_text"]

        outputs.append({
            "_id": ex["_id"],
            "domain": args.domain,
            "question": question,
            "answer": answer,
            "contexts": contexts
        })

    write_jsonl(args.output, outputs)
    print(f"✅ Done. Wrote {len(outputs)} examples.")

if __name__ == "__main__":
    main()
