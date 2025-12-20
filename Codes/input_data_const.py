import json
import os
import argparse


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("-d", "--max_docs", type=int, default=5)
    return p.parse_args()


def is_relevant(ctx, conv):
    if "feedback" not in ctx:
        return False
    fb = ctx["feedback"]["relevant"]
    if "editor" in conv and conv["editor"] in fb:
        return fb[conv["editor"]]["value"] == "yes"
    return fb[conv["author"]]["value"] == "yes"


if __name__ == "__main__":
    args = parse_args()
    conversations = read_json(args.input)

    rows = []

    for conv in conversations:
        conv_id = conv.get("conversation_id", conv["author"])

        history_questions = []
        pending_q = None
        turn_idx = 0

        for msg in conv["messages"]:
            # User turn
            if msg["speaker"] == "user":
                pending_q = msg["text"]
                turn_idx += 1

            # Agent turn (corresponds to current_question)
            elif msg["speaker"] == "agent" and pending_q:
                expected_answer = msg["text"]

                retrieved_docs = []
                for ctx in msg.get("contexts", []):
                    if is_relevant(ctx, conv):
                        retrieved_docs.append({
                            "doc_id": len(retrieved_docs) + 1,
                            "text": ctx["text"]
                        })

                rows.append({
                    "_id": f"{conv_id}<::>{turn_idx}",
                    "history": history_questions.copy(),
                    "current_question": pending_q,
                    "expected_answer": expected_answer,
                    "retrieved_documents": retrieved_docs[:args.max_docs]
                })

                # update history AFTER emitting example
                history_questions.append(pending_q)
                pending_q = None

    out_file = os.path.join(
        args.output,
        "rewrite_inputs_multiturn.jsonl"
    )
    write_jsonl(out_file, rows)
