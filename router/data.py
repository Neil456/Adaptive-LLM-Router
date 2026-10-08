"""MMLU loading, stratified sampling and prompt formatting."""
import pandas as pd
from datasets import load_dataset

from router.config import QUESTIONS_PER_SUBJECT, SEED

# Subject -> super-category, following the grouping in Hendrycks et al. (2021),
# "Measuring Massive Multitask Language Understanding" (categories.py in their repo).
_CATEGORY_SUBJECTS = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics", "high_school_physics",
        "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy",
    ],
    "Other": [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
        "human_aging", "management", "marketing", "medical_genetics", "miscellaneous",
        "nutrition", "professional_accounting", "professional_medicine", "virology",
    ],
}
SUBJECT_TO_CATEGORY = {s: cat for cat, subjects in _CATEGORY_SUBJECTS.items() for s in subjects}
CATEGORIES = list(_CATEGORY_SUBJECTS)


def load_mmlu_sample(per_subject: int = QUESTIONS_PER_SUBJECT, seed: int = SEED) -> pd.DataFrame:
    """Sample `per_subject` questions from each of the 57 MMLU test subjects."""
    ds = load_dataset("cais/mmlu", "all", split="test").to_pandas()
    ds["mmlu_idx"] = ds.index
    sample = (
        ds.groupby("subject", group_keys=False)
        .sample(n=per_subject, random_state=seed)
        .sort_values("mmlu_idx")
        .reset_index(drop=True)
    )
    sample["category"] = sample["subject"].map(SUBJECT_TO_CATEGORY)
    assert sample["category"].notna().all(), "unmapped MMLU subject"
    sample["choices"] = sample["choices"].map(list)
    sample.insert(0, "qid", range(len(sample)))
    sample["prompt"] = sample.apply(format_prompt, axis=1)
    return sample


def format_prompt(row) -> str:
    """Zero-shot MMLU prompt in the style of lm-evaluation-harness."""
    subject = row["subject"].replace("_", " ")
    lines = [f"The following is a multiple choice question about {subject}.", "", row["question"].strip()]
    lines += [f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"])]
    lines.append("Answer:")
    return "\n".join(lines)
