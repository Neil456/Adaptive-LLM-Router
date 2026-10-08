#!/usr/bin/env bash
# Runs the full pipeline. Step 1 is the slow one (~1 hour on a 4-core CPU) and is resumable;
# skip it with SKIP_LLMS=1 to rebuild everything else from the committed data/llm_outputs.csv.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

if [[ "${SKIP_LLMS:-0}" != "1" ]]; then
  $PY -m router.run_llms        # 1. small + large LLM answers, routing labels
fi
$PY -m router.llm_summary       # 2. accuracy summary, Figure 1
$PY -m router.features          # 3. features, embeddings, train/test split
$PY -m router.supervised        # 4. LR / RF / MLP comparison, Figure 2
$PY -m router.clustering        # 5. PCA + K-Means, Figures 3 and 4
$PY -m router.routing           # 6. routing evaluation, Figure 5
