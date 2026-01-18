# cqr.py
# Query rewriting / expansion for MT-RAG eval:
# - Uses current question + prior conversation turns (including prior agent answers)
# - DOES NOT use retrieved contexts (not available in eval input)
# - Topic hints extracted ONLY from user turns to avoid entity drift
# - Generates keyword query via FLAN-T5 + safety validators

import json
import argparse
import os
import re
import torch
import spacy
from typing import List, Dict, Any, Iterable, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ========================
# spaCy
# ========================
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")


# ========================
# Model
# ========================
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)


# ========================
# IO
# ========================
def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ========================
# Helpers
# ========================
def normalize(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    return re.sub(r"\s+", " ", x.strip())


def iter_history_strings(history: Any, only_speaker: Optional[str] = None) -> Iterable[str]:
    """
    history is a list of dicts like:
      {"speaker":"user"/"agent", "text":"..."}
    """
    if not history:
        return
    if not isinstance(history, list):
        history = [history]

    for h in history:
        if isinstance(h, str):
            s = normalize(h)
            if s:
                yield s
            continue

        if isinstance(h, dict):
            if only_speaker and h.get("speaker") != only_speaker:
                continue
            s = normalize(h.get("text", ""))
            if s:
                yield s
            continue

        s = normalize(h)
        if s:
            yield s


def history_tail(history: Any, k: int = 8, only_speaker: Optional[str] = None) -> List[str]:
    hs = list(iter_history_strings(history, only_speaker=only_speaker))
    return hs[-k:]


def simple_tokens(text: Any) -> List[str]:
    return re.findall(r"[a-z0-9\-]+", normalize(text).lower())


# ========================
# Rewrite gating
# ========================
PRONOUNS = {"it", "this", "that", "they", "those", "these", "one", "he", "she", "him", "her", "his", "its"}
ELLIPSIS_STARTERS = ("which ", "or ", "then ", "so ", "what about ", "how about ", "is that ", "is this ")


def contains_pronoun(q: str) -> bool:
    q = normalize(q)
    if not q:
        return False
    doc = nlp(q)
    return any(tok.text.lower() in PRONOUNS for tok in doc)


def is_short_or_fragment(q: str) -> bool:
    q = normalize(q)
    toks = q.split()
    if len(toks) <= 4:
        return True
    if len(toks) <= 3 and not q.endswith("?"):
        return True
    return False


def is_ellipsis(q: str) -> bool:
    ql = normalize(q).lower()
    return ql.startswith(ELLIPSIS_STARTERS) or (ql.endswith("?") and len(ql.split()) <= 4)


def needs_rewrite(question: str) -> bool:
    q = normalize(question)
    if not q:
        return False
    if is_short_or_fragment(q):
        return True
    if contains_pronoun(q):
        return True
    if is_ellipsis(q):
        return True
    return False


# ========================
# Topic hint extraction
# ========================
STOP_HEADS = {"thing", "things", "something", "someone", "stuff"}
USEFUL_ENTITY_LABELS = {"ORG", "GPE", "EVENT", "WORK_OF_ART", "PRODUCT", "PERSON"}


def extract_keywords_from_user_history(history: Any, top_k: int = 4) -> List[str]:
    """
    Extract topic hints ONLY from prior user turns.
    This avoids drifting into agent-introduced entities.
    """
    hs = history_tail(history, k=10, only_speaker="user")
    keywords: List[str] = []

    for h in hs:
        doc = nlp(h)

        for ent in doc.ents:
            if ent.label_ in USEFUL_ENTITY_LABELS:
                keywords.append(ent.text.lower())

        for chunk in doc.noun_chunks:
            head = chunk.root.text.lower()
            if (
                len(chunk.text.split()) >= 2
                and head not in STOP_HEADS
                and not chunk.text.lower().startswith(("what", "which", "how", "why"))
            ):
                keywords.append(chunk.text.lower())

    freq: Dict[str, int] = {}
    for k in keywords:
        freq[k] = freq.get(k, 0) + 1

    ranked = sorted(freq.items(), key=lambda x: -x[1])
    return [k for k, _ in ranked[:top_k]]


def extract_metadata_hints(metadata: Any, max_items: int = 8) -> List[str]:
    if not metadata:
        return []

    out: List[str] = []
    if isinstance(metadata, dict):
        stack = list(metadata.values())
        while stack:
            x = stack.pop()
            if x is None:
                continue
            if isinstance(x, str):
                s = normalize(x)
                if s and len(s.split()) <= 8:
                    out.append(s.lower())
            elif isinstance(x, dict):
                stack.extend(list(x.values()))
            elif isinstance(x, list):
                stack.extend(x)

    out = list(dict.fromkeys(out))
    return out[:max_items]


# ========================
# Prompt
# ========================
BAD_OUTPUT_PATTERNS = (
    "do not invent facts",
    "no repetition",
    "lowercase",
    "space-separated",
    "output only keywords",
    "no explanations",
    "instructions:",
)


def build_prompt(
    question: str,
    history: Any,
    collection: str,
    topic_hints: List[str],
    metadata_hints: List[str],
) -> str:
    # Provide full history (user+agent) to help resolve pronouns/ellipsis
    hist = history_tail(history, k=10, only_speaker=None)
    history_text = "\n".join(f"- {h}" for h in hist) if hist else "- none"

    topic_text = "\n".join(f"- {t}" for t in topic_hints) if topic_hints else "- none"
    meta_text = "\n".join(f"- {m}" for m in metadata_hints) if metadata_hints else "- none"

    return f"""
You rewrite user questions into SHORT keyword queries for document retrieval.

Collection: {collection}

Current question:
{question}

Conversation history (previous turns, user and agent answers):
{history_text}

Topic hints (from prior user questions):
{topic_text}

Metadata hints (weak, may be noisy):
{meta_text}

Hard constraints:
- output ONLY a keyword query, NOT a sentence
- do NOT include instructions or meta text
- do NOT invent new entities unless present in question/history/hints
- resolve pronouns and ellipsis using history
- remove filler words
- no repetition
- lowercase
- space-separated tokens (hyphen allowed inside a token)
- max 12 tokens
""".strip()


# ========================
# Generation + validation
# ========================
def flan_generate(prompt: str, max_tokens: int = 64) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=4,
            do_sample=False,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def postprocess(raw: str, max_terms: int = 12) -> str:
    toks = simple_tokens(raw)
    toks = list(dict.fromkeys(toks))
    return " ".join(toks[:max_terms]).strip()


def is_bad_output(expanded: str) -> bool:
    if not expanded:
        return True
    e = expanded.strip().lower()
    for p in BAD_OUTPUT_PATTERNS:
        if p in e:
            return True
    if len(e) < 3:
        return True
    if len(e.split()) == 1 and e in {"yes", "no", "maybe", "none"}:
        return True
    return False


def safe_rule_fallback(question: str, history: Any) -> str:
    """
    Conservative fallback:
    - keep original tokens
    - append most frequent topic hint if short/pronoun
    """
    q = normalize(question).lower()
    topic_hints = extract_keywords_from_user_history(history, top_k=3)
    topic = topic_hints[0] if topic_hints else ""

    if (is_short_or_fragment(q) or contains_pronoun(q)) and topic:
        return postprocess(f"{q} {topic}")

    return postprocess(q)


# ========================
# Main rewrite logic
# ========================
def rewrite_example(ex: Dict[str, Any]) -> Dict[str, Any]:
    question = normalize(ex.get("current_question", ""))
    history = ex.get("history", []) or []
    metadata = ex.get("metadata", {}) or {}
    collection = ex.get("collection") or "unknown"

    if not question:
        return {
            "_id": ex.get("_id"),
            "collection": collection,
            "original_question": "",
            "expanded_query": "",
            "rewrite_applied": False,
        }

    if not needs_rewrite(question):
        return {
            "_id": ex.get("_id"),
            "collection": collection,
            "original_question": question,
            "expanded_query": question,
            "rewrite_applied": False,
        }

    topic_hints = extract_keywords_from_user_history(history, top_k=4)
    metadata_hints = extract_metadata_hints(metadata, max_items=8)

    prompt = build_prompt(
        question=question,
        history=history,
        collection=collection,
        topic_hints=topic_hints,
        metadata_hints=metadata_hints,
    )

    raw = flan_generate(prompt)
    expanded = postprocess(raw)

    if is_bad_output(expanded):
        expanded = safe_rule_fallback(question, history)

    if not expanded:
        expanded = postprocess(question)

    return {
        "_id": ex.get("_id"),
        "collection": collection,
        "original_question": question,
        "expanded_query": expanded,
        "rewrite_applied": True,
    }


# ========================
# CLI
# ========================
def main():
    parser = argparse.ArgumentParser("CQR Query rewriting (FLAN-T5) for MT-RAG eval")
    parser.add_argument("-i", "--input", required=True, help="input jsonl from dataset.py")
    parser.add_argument("-o", "--output", required=True, help="output rewritten jsonl")
    args = parser.parse_args()

    outputs: List[Dict[str, Any]] = []
    for ex in read_jsonl(args.input):
        outputs.append(rewrite_example(ex))

    write_jsonl(args.output, outputs)
    print("✅ Done — CQR rewrite")


if __name__ == "__main__":
    main()
