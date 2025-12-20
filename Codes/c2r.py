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
            f.write(json.dumps(r) + "\n")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", required=True)
    p.add_argument("-o", "--output", required=True)
    p.add_argument(
        "-m", "--mode",
        choices=["q_only", "q_a", "q_a_d"],
        required=True
    )
    p.add_argument("-d", "--max_docs", type=int, default=2)
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

        history = []
        pending_q = None
        turn_idx = 0

        for msg in conv["messages"]:
            if msg["speaker"] == "user":
                pending_q = msg["text"]
                turn_idx += 1

            elif msg["speaker"] == "agent" and pending_q:
                answer = msg["text"]
                contexts = []

                for ctx in msg.get("contexts", []):
                    if is_relevant(ctx, conv):
                        contexts.append({
                            "doc": len(contexts) + 1,
                            "text": ctx["text"]
                        })

                history.append({
                    "turn": turn_idx,
                    "question": pending_q,
                    "answer": answer,
                    "contexts": contexts[:args.max_docs]
                })

                # build history according to mode
                structured_history = []
                for h in history[:-1]:
                    item = {
                        "turn": h["turn"],
                        "question": h["question"]
                    }
                    if args.mode in ["q_a", "q_a_d"]:
                        item["answer"] = h["answer"]
                    if args.mode == "q_a_d":
                        item["contexts"] = h["contexts"]
                    structured_history.append(item)

                rows.append({
                    "_id": f"{conv_id}<::>{turn_idx}",
                    "mode": args.mode,
                    "history": structured_history,
                    "current_question": history[-1]["question"]
                })

                pending_q = None

    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(
        args.output,
        f"rewrite_inputs_structured_{args.mode}.jsonl"
    )
    write_jsonl(out_file, rows)
