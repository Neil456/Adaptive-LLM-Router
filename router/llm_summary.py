"""Step 2: summarize small vs large model accuracy and the routing labels (Figure 1).

Usage: python -m router.llm_summary
"""
import json

import numpy as np
import pandas as pd

from router.config import LLM_META, LLM_OUTPUTS, RESULTS_DIR
from router.data import CATEGORIES
from router.plotting import AXIS, INK_2, MUTED, SERIES, apply_style, bar, plt, save

OUTCOMES = {
    "Both correct": (1, 1),
    "Only small correct": (1, 0),
    "Only large correct\n(label = LARGE)": (0, 1),
    "Both wrong": (0, 0),
}


def ci95(p: float, n: int) -> float:
    """Half-width of a normal-approximation 95% confidence interval for a proportion."""
    return 1.96 * np.sqrt(p * (1 - p) / n)


def main():
    df = pd.read_csv(LLM_OUTPUTS)
    meta = json.loads(LLM_META.read_text())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    groups = [(c, df[df["category"] == c]) for c in CATEGORIES] + [("All", df)]
    by_cat = pd.DataFrame([{
        "category": name,
        "n": len(g),
        "small_acc": g["small_correct"].mean(),
        "mid_acc": g["mid_correct"].mean(),
        "large_acc": g["large_correct"].mean(),
        "large_label_rate": g["label"].mean(),
    } for name, g in groups])
    by_cat.to_csv(RESULTS_DIR / "accuracy_by_category.csv", index=False, float_format="%.4f")

    by_subject = (
        df.groupby(["category", "subject"])
        .agg(n=("qid", "size"), small_acc=("small_correct", "mean"), mid_acc=("mid_correct", "mean"),
             large_acc=("large_correct", "mean"), large_label_rate=("label", "mean"))
        .reset_index()
        .sort_values("large_label_rate", ascending=False)
    )
    by_subject.to_csv(RESULTS_DIR / "accuracy_by_subject.csv", index=False, float_format="%.4f")

    outcomes = {
        name.replace("\n", " "): int(((df["small_correct"] == s) & (df["large_correct"] == l)).sum())
        for name, (s, l) in OUTCOMES.items()
    }
    summary = {
        "n_questions": len(df),
        **{f"{size}_model": m["model"] for size, m in meta.items()},
        **{f"{size}_params": m["n_params"] for size, m in meta.items()},
        **{f"{size}_acc": df[f"{size}_correct"].mean() for size in meta},
        **{f"{size}_mean_latency_s": df[f"{size}_latency_s"].mean() for size in meta},
        "mean_prompt_tokens": df["n_tokens"].mean(),
        "outcomes": outcomes,
        "large_label_rate": df["label"].mean(),
    }
    (RESULTS_DIR / "llm_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(by_cat.to_string(index=False))
    print(json.dumps(summary, indent=2))

    plot(by_cat, df)


def plot(by_cat: pd.DataFrame, df: pd.DataFrame):
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})

    x = np.arange(len(by_cat))
    w = 0.26
    for offset, col, label, color in [(-w, "small_acc", "Small: Qwen2.5-0.5B", SERIES[0]),
                                      (0, "mid_acc", "Qwen2.5-3B (v1 large)", SERIES[6]),
                                      (w, "large_acc", "Large: Qwen2.5-7B", SERIES[1])]:
        err = [ci95(p, n) for p, n in zip(by_cat[col], by_cat["n"])]
        bar(ax1, x + offset, by_cat[col], color, w, label=label)
        ax1.errorbar(x + offset, by_cat[col], yerr=err, fmt="none", ecolor=INK_2, elinewidth=1, capsize=0)
        overall = by_cat[col].iloc[-1]
        ax1.text(x[-1] + offset, overall + err[-1] + 0.015, f"{overall:.0%}", ha="center", va="bottom",
                 fontsize=8.5, color=INK_2)
    ax1.axhline(0.25, color=MUTED, linewidth=1)
    ax1.text(-0.66, 0.255, "chance", color=MUTED, fontsize=8, va="bottom")
    ax1.set_xlim(-0.7, len(x) - 0.5)
    ax1.set_xticks(x, [c.replace(" ", "\n") for c in by_cat["category"]])
    ax1.set_ylim(0, 1)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_ylabel("QA accuracy")
    ax1.set_title("a) Accuracy by MMLU category (95% CI)")
    ax1.grid(axis="x", visible=False)
    ax1.legend(loc="upper left", ncols=3, fontsize=8.5, handlelength=1.2, columnspacing=1.0)

    names = list(OUTCOMES)
    shares = [((df["small_correct"] == s) & (df["large_correct"] == l)).mean() for s, l in OUTCOMES.values()]
    y = np.arange(len(names))[::-1]
    colors = [SERIES[1] if "LARGE" in n else AXIS for n in names]
    ax2.barh(y, shares, height=0.5, color=colors, edgecolor="none")
    for yi, s in zip(y, shares):
        ax2.text(s + 0.01, yi, f"{s:.1%}", va="center", fontsize=9, color=INK_2)
    ax2.set_yticks(y, names)
    ax2.set_xlim(0, max(shares) * 1.25)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xlabel("Share of questions")
    ax2.set_title("b) Outcome per question")
    ax2.grid(axis="y", visible=False)

    fig.tight_layout()
    save(fig, "fig1_small_vs_large_accuracy.png")


if __name__ == "__main__":
    main()
