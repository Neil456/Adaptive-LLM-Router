# Adaptive LLM Router

Train a classifier that decides, per question, whether a **small LLM** is good enough or the
question should go to a **larger LLM**, then measure the accuracy/compute trade-off of
routing with it.

- **Small model:** `Qwen/Qwen2.5-0.5B-Instruct` (0.49B parameters)
- **Large model:** `Qwen/Qwen2.5-3B-Instruct` (3.09B parameters)
- **Data:** 5,700 MMLU test questions (100 from each of the 57 subjects)
- **Label:** `LARGE` if the small model is wrong and the large model is right, else `SMALL`

The full write-up, with all figures and tables, is in **[REPORT.md](REPORT.md)**.

## Results at a glance (test split, 1,140 questions)

| Policy | QA accuracy | Sent to large | Compute saved vs always-large |
|---|---:|---:|---:|
| Always small | 47.6% | 0% | 84% |
| Always large | 67.5% | 100% | 0% |
| Random routing | 53.2% | 28% | 61% |
| Learned router (prompt features) | 54.8% | 29% | 60% |
| Learned router (+ small-model confidence) | 56.3% | 28% | 58% |
| Oracle (upper bound) | 75.3% | 28% | 62% |

- **Prompt features alone barely predict the label:** Random Forest reaches a test ROC-AUC of
  0.58, with a cross-validated AUC of 0.53.
- **Adding the small model's confidence helps:** test ROC-AUC rises to 0.66, and the router
  beats random routing by 3.2 points (95% CI +1.5 to +4.9).
- **K-Means clusters of the prompt embeddings differ in how often they need the large model**
  (22.7% to 30.8%, χ² p = 0.003). Mid-difficulty topics such as economics need it most; the
  hardest topics often defeat both models.

## Layout

```
router/
  config.py        paths, model names, constants
  data.py          MMLU sampling, category mapping, prompt format
  run_llms.py      step 1: run both LLMs on every question -> data/llm_outputs.csv
  llm_summary.py   step 2: small vs large accuracy            -> Figure 1
  features.py      step 3: features, embeddings, train/test split
  supervised.py    step 4: Logistic Regression / Random Forest / MLP -> Figure 2
  clustering.py    step 5: PCA + K-Means on embeddings        -> Figures 3, 4
  routing.py       step 6: routing baselines vs learned router -> Figure 5
  plotting.py      shared figure style
data/              LLM outputs (committed), features, split
results/           metrics tables (CSV / JSON)
figures/           the five report figures
```

## Running it

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Everything except the LLM runs, using the committed data/llm_outputs.csv (~5 min on CPU)
SKIP_LLMS=1 ./run_pipeline.sh

# Full pipeline including the LLM runs (~1 hour on a 4-core CPU, resumable)
./run_pipeline.sh
```

Each step can also be run on its own, e.g. `python -m router.supervised`.

The LLMs answer by next-token scoring: one forward pass per question, reading the logits of
` A`/` B`/` C`/` D` after `Answer:`. No sampling is involved, so runs are deterministic up to
floating-point differences between CPUs.
