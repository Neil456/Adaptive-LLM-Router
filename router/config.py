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

SMALL_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LARGE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LABEL_NAMES = {0: "SMALL", 1: "LARGE"}
