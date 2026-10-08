"""Step 1: run every sampled MMLU question through the small and the large LLM.

Each model answers by next-token scoring: we read the logits of the tokens " A", " B",
" C" and " D" right after "Answer:" and pick the highest. This needs one forward pass per
question (no sampling), and the softmax over the four letters doubles as a confidence score.

Usage: python -m router.run_llms [--limit N]
"""
import argparse
import json
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from router.config import CACHE_DIR, LARGE_MODEL, LLM_META, LLM_OUTPUTS, SMALL_MODEL
from router.data import load_mmlu_sample

CHECKPOINT_EVERY = 100


def score_model(model_name: str, questions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Answer every question with `model_name`, resuming from a cached partial run if present."""
    cache = CACHE_DIR / f"{model_name.split('/')[-1]}.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    todo = questions[~questions["qid"].isin(done.get("qid", []))]
    print(f"[{model_name}] {len(done)} cached, {len(todo)} to run")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    letter_ids = [tokenizer.encode(" " + letter)[0] for letter in "ABCD"]

    rows = []
    with torch.inference_mode():
        for i, q in enumerate(todo.itertuples(), 1):
            enc = tokenizer(q.prompt, return_tensors="pt")
            start = time.perf_counter()
            logits = model(**enc).logits[0, -1, letter_ids].float()
            latency = time.perf_counter() - start
            probs = torch.softmax(logits, dim=-1).numpy()
            rows.append({
                "qid": q.qid,
                "n_tokens": enc["input_ids"].shape[1],
                "pred": int(probs.argmax()),
                **{f"p_{letter}": float(p) for letter, p in zip("ABCD", probs)},
                "latency_s": latency,
            })
            if i % CHECKPOINT_EVERY == 0 or i == len(todo):
                done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
                done.to_csv(cache, index=False)
                rows = []
                print(f"[{model_name}] {len(done)}/{len(questions)}", flush=True)

    meta = {"model": model_name, "n_params": n_params, "mean_latency_s": float(done["latency_s"].mean())}
    return done.sort_values("qid").reset_index(drop=True), meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions (smoke test)")
    args = parser.parse_args()

    torch.manual_seed(0)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    questions = load_mmlu_sample()
    if args.limit:
        questions = questions.head(args.limit)

    out = questions[["qid", "mmlu_idx", "subject", "category", "question", "choices", "answer", "prompt"]].copy()
    out["choices"] = out["choices"].map(json.dumps)
    meta = {}
    for size, model_name in [("small", SMALL_MODEL), ("large", LARGE_MODEL)]:
        scored, meta[size] = score_model(model_name, questions)
        scored = scored[scored["qid"].isin(out["qid"])].set_index("qid").loc[out["qid"]].reset_index()
        out["n_tokens"] = scored["n_tokens"].to_numpy()  # same Qwen2.5 tokenizer for both models
        out[f"{size}_pred"] = scored["pred"].to_numpy()
        out[f"{size}_correct"] = (scored["pred"].to_numpy() == out["answer"].to_numpy()).astype(int)
        probs = scored[[f"p_{letter}" for letter in "ABCD"]].to_numpy()
        out[f"{size}_conf"] = probs.max(axis=1)
        out[f"{size}_latency_s"] = scored["latency_s"].to_numpy()
        for letter, col in zip("ABCD", probs.T):
            out[f"{size}_p_{letter}"] = col

    # Routing label: LARGE only when escalating actually fixes the answer.
    # If both models are wrong the large model does not help, so the cheap model is the right call.
    out["label"] = ((out["small_correct"] == 0) & (out["large_correct"] == 1)).astype(int)

    out.to_csv(LLM_OUTPUTS, index=False, float_format="%.6g")
    LLM_META.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"small acc={out['small_correct'].mean():.3f}  large acc={out['large_correct'].mean():.3f}  "
          f"LARGE label rate={out['label'].mean():.3f}  -> {LLM_OUTPUTS}")


if __name__ == "__main__":
    main()
