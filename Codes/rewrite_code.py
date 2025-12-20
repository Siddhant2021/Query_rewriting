import json
import argparse
import os
import re
import spacy
from collections import Counter

# ------------------------
# spaCy
# ------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# ------------------------
# IO
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
# Constants
# ------------------------
PRONOUNS = {"it", "this", "that", "they", "those", "these", "one"}

ELLIPSIS_STARTERS = (
    "which ", "or ", "then ", "so ",
    "what about ", "how about ", "is that ", "is this "
)

STOP_HEADS = {
    "thing", "things", "something", "someone", "stuff"
}

PREPOSITIONS = {"in", "on", "of", "to", "for"}

# ------------------------
# Helpers
# ------------------------


def normalize(text):
    return re.sub(r"\s+", " ", text.strip())


def iter_history_text(history):
    for turn in history:
        if isinstance(turn, dict):
            yield turn.get("question") or turn.get("Question") or ""
        elif isinstance(turn, str):
            yield turn
        else:
            continue


def contains_pronoun(q):
    return any(tok.text.lower() in PRONOUNS for tok in nlp(q))


def is_ellipsis(q):
    ql = q.lower()
    return (
        ql.startswith(ELLIPSIS_STARTERS)
        or (ql.endswith("?") and len(ql.split()) <= 4)
    )


def is_fragment(q):
    return len(q.split()) <= 3 and not q.endswith("?")

# ------------------------
# GuideCQR keyword filtering (conversation-only)
# ------------------------


def extract_keywords_from_history(history, top_k=4):
    keywords = []

    for text in iter_history_text(history):
        if not text:
            continue

        doc = nlp(text)

        # Named entities
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "GPE", "EVENT", "WORK_OF_ART"}:
                keywords.append(ent.text)

        # Noun chunks
        for chunk in doc.noun_chunks:
            head = chunk.root.text.lower()
            if (
                len(chunk.text.split()) >= 2
                and head not in STOP_HEADS
                and not chunk.text.lower().startswith(
                    ("what", "which", "how", "why")
                )
            ):
                keywords.append(chunk.text)

    counter = Counter(keywords)
    return [k for k, _ in counter.most_common(top_k)]

# ------------------------
# NEW: Fragment expansion (SAFE)
# ------------------------


def expand_fragment(fragment, topic):
    frag = fragment.lower()

    # Prepositional fragments: "in water"
    if frag.split()[0] in PREPOSITIONS:
        return f"{topic} {fragment}"

    # Single noun: "Pollution", "Solutions"
    return f"{fragment} of {topic}"

# ------------------------
# Rewrite logic
# ------------------------


def rewrite_query(ex):
    q = normalize(ex["current_question"])
    history = ex.get("history", [])

    keywords = extract_keywords_from_history(history)
    topic = keywords[0] if keywords else None

    # ----------------
    # Fragment expansion (NEW)
    # ----------------
    if is_fragment(q) and topic:
        rewritten = expand_fragment(q, topic)
        return normalize(rewritten), True

    # ----------------
    # Standalone question
    # ----------------
    if not contains_pronoun(q) and not is_ellipsis(q):
        return q, False

    # ----------------
    # Pronoun resolution
    # ----------------
    if contains_pronoun(q) and topic:
        rewritten = re.sub(
            r"\bit\b|\bthis\b|\bthat\b",
            topic,
            q,
            flags=re.I
        )
        return normalize(rewritten), True

    # ----------------
    # Ellipsis completion
    # ----------------
    if is_ellipsis(q) and len(keywords) >= 2:
        a, b = keywords[0], keywords[1]
        ql = q.lower()

        if ql.startswith("which"):
            return f"Which is more important, {a} or {b}?", True

        if ql.startswith("or"):
            return f"Is {b} easier than {a}?", True

        if ql.startswith(("what about", "how about")):
            return f"What about {b} compared to {a}?", True

    # ----------------
    # Fallback
    # ----------------
    return q, False

# ------------------------
# Main
# ------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GuideCQR + Safe Fragment Expansion"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    outputs = []

    for ex in read_jsonl(args.input):
        rewritten, applied = rewrite_query(ex)
        outputs.append({
            "_id": ex.get("_id"),
            "original_question": ex["current_question"],
            "rewritten_query": rewritten,
            "rewrite_applied": applied
        })

    write_jsonl(args.output, outputs)
    print(" Done — GuideCQR + fragment expansion")


if __name__ == "__main__":
    main()
