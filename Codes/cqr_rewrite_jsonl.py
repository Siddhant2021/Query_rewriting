import json
import argparse
import os
import re
import torch
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter

# ========================
# spaCy
# ========================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# ========================
# FLAN-T5
# ========================
MODEL_NAME = "google/flan-t5-base"  # upgrade to large/xl if GPU allows

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

# ========================
# IO
# ========================


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

# ========================
# Rewrite gating (IMPORTANT)
# ========================


PRONOUNS = {"it", "this", "that", "they", "those", "these", "one"}


def normalize(text):
    return re.sub(r"\s+", " ", text.strip())


def contains_pronoun(q):
    return any(tok.text.lower() in PRONOUNS for tok in nlp(q))


def needs_rewrite(question):
    q = question.lower().strip()
    tokens = q.split()

    # short fragments MUST rewrite
    if len(tokens) <= 4:
        return True

    if contains_pronoun(q):
        return True

    if q.startswith(("what about", "how about", "or ")):
        return True

    return False

# ========================
# Retrieval context extraction
# ========================


def extract_retrieval_hints(ex, top_k=3):
    docs = ex.get("retrieved_documents", [])[:top_k]

    titles = []
    entities = []

    for d in docs:
        title = d.get("title")
        if title:
            titles.append(title.lower())

        text = d.get("text", "")
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ in {"ORG", "GPE", "EVENT", "WORK_OF_ART", "PRODUCT"}:
                entities.append(ent.text.lower())

        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) >= 2:
                entities.append(chunk.text.lower())

    # dedupe + trim
    titles = list(dict.fromkeys(titles))[:5]
    entities = list(dict.fromkeys(entities))[:8]

    return titles, entities

# ========================
# Prompt builder
# ========================


def build_prompt(question, history, titles, entities):
    history_text = "\n".join(f"- {h}" for h in history if h)

    titles_text = "\n".join(f"- {t}" for t in titles) if titles else "- none"
    entities_text = "\n".join(
        f"- {e}" for e in entities) if entities else "- none"

    return f"""
You generate search queries for an information retrieval system.

Current user question:
{question}

Conversation history:
{history_text}

Retrieved context hints (from previous search, may be partial):
Document titles:
{titles_text}

Key concepts:
{entities_text}

Instructions:
- resolve references using conversation history
- ground the query in the retrieved concepts
- output ONLY keywords, not sentences
- no explanations
- no repetition
- lowercase
- space-separated terms
- do NOT invent facts
"""

# ========================
# FLAN inference
# ========================


def flan_generate(prompt, max_tokens=48):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=4,
            do_sample=False,
            early_stopping=True
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# ========================
# Postprocess (ELSER-safe)
# ========================


def postprocess(text, max_terms=12):
    tokens = re.findall(r"[a-z0-9\-]+", text.lower())
    tokens = list(dict.fromkeys(tokens))
    return " ".join(tokens[:max_terms])

# ========================
# Rewrite logic
# ========================


def rewrite_example(ex):
    question = normalize(ex["current_question"])
    history = ex.get("history", [])

    if not needs_rewrite(question):
        return {
            "_id": ex.get("_id"),
            "original_question": question,
            "expanded_query": question,
            "rewrite_applied": False
        }

    titles, entities = extract_retrieval_hints(ex)
    prompt = build_prompt(question, history, titles, entities)

    raw = flan_generate(prompt)
    expanded = postprocess(raw)

    if not expanded:
        expanded = question.lower()

    return {
        "_id": ex.get("_id"),
        "original_question": question,
        "expanded_query": expanded,
        "rewrite_applied": True
    }

# ========================
# OPTIONAL API HOOK (COMMENTED)
# ========================


"""
def llm_generate_api(prompt):
    payload = {
        "model": "gpt-oss-120b",
        "prompt": prompt,
        "temperature": 0.0
    }
    response = requests.post(API_URL, json=payload)
    return response.json()["text"]
"""

# ========================
# CLI
# ========================


def main():
    parser = argparse.ArgumentParser("Retrieval-aware CQR (FLAN-T5)")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    outputs = []
    for ex in read_jsonl(args.input):
        outputs.append(rewrite_example(ex))

    write_jsonl(args.output, outputs)
    print("✅ Done — Retrieval-aware CQR")


if __name__ == "__main__":
    main()
