import json
from typing import List, Dict, Any


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_qr_dataset_from_messages(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build query-rewriting dataset from your EXACT input format.

    Rules:
    - history = previous USER questions only
    - agent answers are NOT part of history
    - metadata = user enrichments
    - expected_answer = next agent message
    - retrieved_documents = contexts from agent message
    """

    conversation_id = conversation["author"]
    messages = conversation["messages"]

    dataset = []
    history: List[str] = []

    i = 0
    turn_id = 1

    while i < len(messages):
        msg = messages[i]

        if msg["speaker"] == "user":
            current_question = msg["text"]
            metadata = msg.get("enrichments", {})

            expected_answer = ""
            retrieved_documents = []

            # Look ahead for agent answer
            if i + 1 < len(messages) and messages[i + 1]["speaker"] == "agent":
                agent_msg = messages[i + 1]
                expected_answer = agent_msg.get("text", "")
                retrieved_documents = agent_msg.get("contexts", [])

            record = {
                "_id": f"{conversation_id}::<{turn_id}>",
                "history": history.copy(),
                "current_question": current_question,
                "metadata": metadata,
                "expected_answer": expected_answer,
                "retrieved_documents": retrieved_documents
            }

            dataset.append(record)

            history.append(current_question)
            turn_id += 1

        i += 1

    return dataset


def main(input_path: str, output_path: str):
    conversations = load_json(input_path)

    final_dataset = []

    for conv in conversations:
        final_dataset.extend(build_qr_dataset_from_messages(conv))

    write_jsonl(output_path, final_dataset)

    print(f"✔ Wrote {len(final_dataset)} records to {output_path}")


if __name__ == "__main__":
    input_path = "human/conversations/conversations.json"
    output_path = "human/rt2/input.jsonl"

    main(input_path, output_path)
