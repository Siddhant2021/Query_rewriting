# dataset.py
import json
from typing import List, Dict, Any


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _infer_collection(conv: Dict[str, Any], default: str = "unknown") -> str:
    """
    Infer corpus collection.
    Eval files often use: "Collection"
    """
    for k in ("Collection", "collection", "dataset", "source", "corpus"):
        if k in conv and isinstance(conv[k], str) and conv[k].strip():
            return conv[k].strip()
    return default


def _infer_conversation_id(conv: Dict[str, Any], default: str = "unknown") -> str:
    """
    Eval files use: conversation_id
    """
    for k in ("conversation_id", "id", "_id", "author"):
        if k in conv and isinstance(conv[k], str) and conv[k].strip():
            return conv[k].strip()
    return default


def build_qr_dataset(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build query rewriting dataset.

    For each USER turn:
      - history includes all prior turns (user+agent)
      - current_question is the current user utterance
      - metadata copied from the user turn if present

    Note: Eval inputs do NOT contain gold contexts/documents, so we do not include them.
    """
    conversation_id = _infer_conversation_id(conversation)
    collection = _infer_collection(conversation)

    # MT-RAG eval uses "input" list
    messages = conversation.get("input") or conversation.get("messages") or []
    if not isinstance(messages, list):
        messages = []

    dataset: List[Dict[str, Any]] = []
    history: List[Dict[str, str]] = []
    turn_id = 1

    for msg in messages:
        speaker = msg.get("speaker")
        text = msg.get("text", "") or ""
        if not isinstance(text, str):
            text = str(text)

        # create record only on user turns
        if speaker == "user":
            metadata = msg.get("metadata", {}) or msg.get("enrichments", {}) or {}
            record = {
                "_id": f"{conversation_id}::<{turn_id}>",
                "collection": collection,
                "history": history.copy(),  # prior turns user+agent
                "current_question": text,
                "metadata": metadata,
            }
            dataset.append(record)
            turn_id += 1

        # always update history for both user+agent
        if text.strip():
            history.append({"speaker": speaker or "unknown", "text": text})

    return dataset


def main(input_path: str, output_path: str):
    conversations = load_json(input_path)
    final_dataset: List[Dict[str, Any]] = []

    for conv in conversations:
        final_dataset.extend(build_qr_dataset(conv))

    write_jsonl(output_path, final_dataset)
    print(f"✔ Wrote {len(final_dataset)} records to {output_path}")


if __name__ == "__main__":
    input_path = "human/conversations/conversations.json"
    output_path = "human/rt3/input.jsonl"
    main(input_path, output_path)
