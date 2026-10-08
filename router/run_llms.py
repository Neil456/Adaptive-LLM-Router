"""Step 1: run every sampled MMLU question through each LLM in `config.MODELS`.

Each model answers by next-token scoring: we read the logits of the tokens " A", " B",
" C" and " D" right after "Answer:" and pick the highest. This needs one forward pass per
question (no sampling), and the softmax over the four letters doubles as a confidence score.

Hugging Face models run in bfloat16 with PyTorch; GGUF models run 4-bit quantized with
llama.cpp (for models too large for RAM in bfloat16).

Usage: python -m router.run_llms [--limit N]
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from router.config import CACHE_DIR, GGUF_DIR, LLM_META, LLM_OUTPUTS, MODELS
from router.data import load_mmlu_sample

CHECKPOINT_EVERY = 100


def load_hf(spec: dict):
    """Return (score(prompt) -> (n_tokens, letter_logits, latency_s), n_params) for a HF model."""
    tokenizer = AutoTokenizer.from_pretrained(spec["name"])
    model = AutoModelForCausalLM.from_pretrained(spec["name"], dtype=torch.bfloat16).eval()
    letter_ids = [tokenizer.encode(" " + letter)[0] for letter in "ABCD"]

    @torch.inference_mode()
    def score(prompt):
        enc = tokenizer(prompt, return_tensors="pt")
        start = time.perf_counter()
        logits = model(**enc).logits[0, -1, letter_ids].float().numpy()
        return enc["input_ids"].shape[1], logits, time.perf_counter() - start

    return score, sum(p.numel() for p in model.parameters())


def load_gguf(spec: dict):
    """Same as `load_hf`, for a quantized GGUF model run with llama.cpp."""
    import llama_cpp
    from huggingface_hub import hf_hub_download

    paths = [hf_hub_download(spec["repo"], f, local_dir=GGUF_DIR) for f in spec["files"]]
    llm = llama_cpp.Llama(model_path=paths[0], n_ctx=2048, n_batch=1024, n_threads=os.cpu_count(),
                          verbose=False)
    letter_ids = [llm.tokenize((" " + letter).encode(), add_bos=False)[0] for letter in "ABCD"]

    def score(prompt):
        tokens = llm.tokenize(prompt.encode(), add_bos=False)
        start = time.perf_counter()
        llm.reset()
        llm.eval(tokens)
        logits = np.ctypeslib.as_array(llama_cpp.llama_get_logits_ith(llm._ctx.ctx, -1), shape=(llm.n_vocab(),))
        return len(tokens), logits[letter_ids].astype(np.float32), time.perf_counter() - start

    return score, llama_cpp.llama_model_n_params(llm.model)


def score_model(spec: dict, questions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Answer every question with one model, resuming from a cached partial run if present."""
    model_name = spec["name"]
    cache = CACHE_DIR / f"{model_name.split('/')[-1]}.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    todo = questions[~questions["qid"].isin(done.get("qid", []))]
    print(f"[{model_name}] {len(done)} cached, {len(todo)} to run")

    score, n_params = (load_gguf if spec["backend"] == "gguf" else load_hf)(spec)
    rows = []
    for i, q in enumerate(todo.itertuples(), 1):
        n_tokens, logits, latency = score(q.prompt)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        rows.append({
            "qid": q.qid,
            "n_tokens": n_tokens,
            "pred": int(probs.argmax()),
            **{f"p_{letter}": float(p) for letter, p in zip("ABCD", probs)},
            "latency_s": latency,
        })
        if i % CHECKPOINT_EVERY == 0 or i == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(cache, index=False)
            rows = []
            print(f"[{model_name}] {len(done)}/{len(questions)}", flush=True)

    meta = {"model": model_name, "backend": spec["backend"], "n_params": int(n_params),
            "mean_latency_s": float(done["latency_s"].mean())}
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
    for size, spec in MODELS.items():
        scored, meta[size] = score_model(spec, questions)
        scored = scored[scored["qid"].isin(out["qid"])].set_index("qid").loc[out["qid"]].reset_index()
        if size == "small":
            out["n_tokens"] = scored["n_tokens"].to_numpy()  # all models share the Qwen2.5 tokenizer
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
    print(f"small acc={out['small_correct'].mean():.3f}  mid acc={out['mid_correct'].mean():.3f}  "
          f"large acc={out['large_correct'].mean():.3f}  "
          f"LARGE label rate={out['label'].mean():.3f}  -> {LLM_OUTPUTS}")


if __name__ == "__main__":
    main()
