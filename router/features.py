"""Step 3: build router features and the train/test split.

Features (all available before the large model is called):
  - prompt_tokens     prompt length in tokens
  - num_numbers       count of numbers in question + choices
  - num_math_symbols  count of math operators / symbols
  - category          MMLU super-category (STEM, Humanities, Social Sciences, Other)
  - emb_*             sentence embedding of question + choices (all-MiniLM-L6-v2, 384-d)
  - small_conf        optional: the small model's probability for its chosen answer

Usage: python -m router.features
"""
import json
import re

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from router.config import EMBEDDING_MODEL, EMBEDDINGS, FEATURES, LLM_OUTPUTS, SEED, SPLIT, TEST_SIZE

NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
# Operators and symbols; a hyphen only counts as minus before a digit or between spaces,
# so hyphenated words ("well-known") are not counted.
MATH_RE = re.compile(r"[+*/=^<>%√∑∫π×÷≤≥≠±∞∈∪∩]|\s[-−]\s|[-−](?=\d)")

NUMERIC_FEATURES = ["prompt_tokens", "num_numbers", "num_math_symbols"]
CATEGORICAL_FEATURES = ["category"]
CONFIDENCE_FEATURES = ["small_conf"]


def question_text(row) -> str:
    """The user-visible content of a prompt: question plus lettered choices (no template header)."""
    choices = json.loads(row["choices"]) if isinstance(row["choices"], str) else row["choices"]
    return "\n".join([row["question"].strip()] + [f"{l}. {c}" for l, c in zip("ABCD", choices)])


def handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df.apply(question_text, axis=1)
    return pd.DataFrame({
        "qid": df["qid"],
        "prompt_tokens": df["n_tokens"],
        "num_numbers": text.map(lambda t: len(NUMBER_RE.findall(t))),
        "num_math_symbols": text.map(lambda t: len(MATH_RE.findall(t))),
        "category": df["category"],
        "small_conf": df["small_conf"],
    })


def embed(df: pd.DataFrame) -> np.ndarray:
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    text = df.apply(question_text, axis=1).tolist()
    return model.encode(text, batch_size=64, normalize_embeddings=True, show_progress_bar=True)


def load_features() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return (features, embeddings, llm_outputs), all row-aligned by qid, with a `split` column."""
    df = pd.read_csv(LLM_OUTPUTS)
    feats = pd.read_csv(FEATURES)
    split = pd.read_csv(SPLIT)
    emb = np.load(EMBEDDINGS)
    assert (feats["qid"].to_numpy() == df["qid"].to_numpy()).all()
    assert len(emb) == len(df)
    feats = feats.merge(split, on="qid", how="left")
    df = df.merge(split, on="qid", how="left")
    return feats, emb, df


def main():
    df = pd.read_csv(LLM_OUTPUTS)
    feats = handcrafted_features(df)
    feats.to_csv(FEATURES, index=False)
    np.save(EMBEDDINGS, embed(df).astype(np.float32))

    train_qid, test_qid = train_test_split(
        df["qid"], test_size=TEST_SIZE, stratify=df["label"], random_state=SEED
    )
    split = pd.concat([
        pd.DataFrame({"qid": train_qid, "split": "train"}),
        pd.DataFrame({"qid": test_qid, "split": "test"}),
    ]).sort_values("qid")
    split.to_csv(SPLIT, index=False)

    print(feats.describe(include="all").T[["mean", "std", "min", "max", "top"]])
    print(split["split"].value_counts())


if __name__ == "__main__":
    main()
