"""Convert allenai/tulu-3-sft-mixture conversations into RWKV-v7 G1x JSONL format.

Output format (one JSONL line per conversation):
    {"text": "User: ...\\n\\nAssistant: <think>\\n</think>\\n\\n...\\n\\nUser: ..."}

Template rules (see RWKV-v7/RWKV7-G1x-templates.txt):
- Turns are joined by "\\n\\n".
- Each turn is "<Role>: <content>" where Role is one of System / User / Assistant.
- Every Assistant turn gets a fake-think prefix "Assistant: <think>\\n</think>\\n\\n<content>"
  (tulu-3 has no explicit think blocks; the fake-think variant is fast and gives nice results).
- Each turn's content is run through clean_txt (collapse blank lines, normalize CRLF, strip).
"""
import argparse
import json
import re

from datasets import load_dataset


_BLANK_LINES_RE = re.compile(r"\n{2,}")


def clean_txt(txt: str) -> str:
    """Per-message content cleaner from RWKV7-G1x-templates.txt."""
    return _BLANK_LINES_RE.sub("\n", txt.replace("\r\n", "\n")).strip()


def format_conversation(messages):
    """Apply RWKV-v7 G1x template to a tulu-3 messages list.

    tulu-3 schema: [{"role": "system" | "user" | "assistant", "content": "..."}, ...]
    """
    parts = []
    for m in messages:
        role = m["role"]
        content = clean_txt(m["content"])
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: <think></think>\n{content}")
        else:
            raise ValueError(f"unknown role: {role!r}")
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True, help="output JSONL path")
    ap.add_argument("--dataset", default="allenai/tulu-3-sft-mixture")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="cap number of rows (smoke testing)")
    ap.add_argument("--messages-field", default="messages",
                    help="name of the messages column (override if dataset uses a different field)")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split=args.split, streaming=False)

    n_written = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if args.limit is not None and n_written >= args.limit:
                break
            messages = row[args.messages_field]
            text = format_conversation(messages)
            f.write(json.dumps({"text": text}, ensure_ascii=False))
            f.write("\n")
            n_written += 1

    print(f"wrote {n_written} conversations to {args.output}")


if __name__ == "__main__":
    main()
