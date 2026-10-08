"""Paths, model names and experiment constants shared by every pipeline step."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

LLM_OUTPUTS = DATA_DIR / "llm_outputs.csv"
LLM_META = DATA_DIR / "llm_meta.json"
FEATURES = DATA_DIR / "features.csv"
EMBEDDINGS = DATA_DIR / "embeddings.npy"
SPLIT = DATA_DIR / "split.csv"

SEED = 42
QUESTIONS_PER_SUBJECT = 100  # every MMLU subject has >= 100 test questions
TEST_SIZE = 0.2

# Each model answers every question. The router chooses between "small" and "large";
# "mid" (the original large model) is kept to show how the size gap affects routing.
# A 7B model does not fit in 15 GB of RAM in bfloat16, so it runs 4-bit quantized via llama.cpp.
MODELS = {
    "small": {"name": "Qwen/Qwen2.5-0.5B-Instruct", "backend": "hf"},
    "mid": {"name": "Qwen/Qwen2.5-3B-Instruct", "backend": "hf"},
    "large": {"name": "Qwen/Qwen2.5-7B-Instruct", "backend": "gguf",
              "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
              "files": ["qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
                        "qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf"]},
}
SMALL_MODEL = MODELS["small"]["name"]
GGUF_DIR = CACHE_DIR / "gguf"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LABEL_NAMES = {0: "SMALL", 1: "LARGE"}
