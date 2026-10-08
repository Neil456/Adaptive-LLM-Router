# Adaptive LLM Router

Train a classifier that decides, per question, whether a **small LLM** is good enough or the
question should go to a **larger LLM**, then measure the accuracy/compute trade-off of
routing with it.

- **Small model:** `Qwen/Qwen2.5-0.5B-Instruct` (0.49B parameters)
- **Large model:** `Qwen/Qwen2.5-7B-Instruct` (7.62B, 4-bit via llama.cpp); `Qwen2.5-3B-Instruct`
  was the large model in the first iteration and is kept for comparison
- **Data:** 5,700 MMLU test questions (100 from each of the 57 subjects)
- **Label:** `LARGE` if the small model is wrong and the large model is right, else `SMALL`

The full write-up (CS 4641/7641 final-report structure, with IEEE references) is in
**[REPORT.md](REPORT.md)**.

## Results at a glance (test split, 1,140 questions)

| Policy | QA accuracy | Sent to 7B | Compute saved vs always-7B |
|---|---:|---:|---:|
| Always small (0.5B) | 47.9% | 0% | 94% |
| Always large (7B) | 72.6% | 100% | 0% |
| Random routing | 56.2% | 34% | 62% |
| Learned router (prompt features) | 58.0% | 36% | 61% |
| **Learned cascade router (+ small-model confidence)** | **60.3%** | 34% | **61%** |
| Learned cascade router, 95%-quality operating point | 70.4% | 72% | 18% |
| Oracle (upper bound) | 80.3% | 32% | 62% |

- **Supervised:** Random Forest is the best of four classifiers (LR, RF, Gradient Boosting,
  MLP). Test ROC-AUC is 0.68 with small-model confidence, versus 0.57 from prompt features
  alone.
- **Routing:**
  - The cascade router beats random routing by **+4.0 points** at the same budget (95% CI
    +2.2 to +6.0).
  - It reaches **APGR 0.62** (0.59 in cross-validation), and recovers half the small→large
    gap with 34% of large-model calls. That is in the range of RouteLLM's routers on MMLU
    (APGR ≈ 0.60, 35%).
- **Unsupervised:** K-Means clusters of prompt embeddings differ in how often they need the
  large model (26.7% to 38.0%, χ² p < 0.001).

## Layout

```
router/
  config.py        paths, model names, constants
  data.py          MMLU sampling, category mapping, prompt format
  run_llms.py      step 1: run all three LLMs on every question -> data/llm_outputs.csv
  llm_summary.py   step 2: small vs large accuracy            -> Figure 1
  features.py      step 3: features, embeddings, train/test split
  supervised.py    step 4: LR / Random Forest / Gradient Boosting / MLP -> Figure 2
  clustering.py    step 5: PCA + K-Means on embeddings        -> Figures 3, 4
  routing.py       step 6: routing baselines vs learned router -> Figure 5
  plotting.py      shared figure style
data/              LLM outputs (committed), features, split
results/           metrics tables (CSV / JSON); results/v1_large_3b/ = first iteration
figures/           the five report figures
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install llama-cpp-python   # only needed to re-run the 7B model (step 1)

# Everything except the LLM runs, using the committed data/llm_outputs.csv (~5 min on CPU)
SKIP_LLMS=1 ./run_pipeline.sh

# Full pipeline including the LLM runs (~4 hours on a 4-core CPU, mostly the 7B; resumable)
./run_pipeline.sh
```

Each step can also be run on its own, e.g. `python -m router.supervised`.

The LLMs answer by next-token scoring: one forward pass per question, reading the logits of
` A`/` B`/` C`/` D` after `Answer:`. No sampling is involved, so runs are deterministic up to
floating-point differences between CPUs. The 7B model runs 4-bit quantized through
llama.cpp because it does not fit in 15 GB of RAM in bfloat16.
