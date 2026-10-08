"""Step 5: unsupervised analysis of prompt embeddings with PCA + K-Means (Figures 3 and 4).

The 384-d sentence embeddings are reduced with PCA and clustered with K-Means for a range of
k. The silhouette score is reported for every k; the chosen k is the best one up to MAX_K,
a cap that keeps the clusters few enough to interpret and to tell apart in a plot. Labels
are never used for fitting; they are only joined afterwards to ask whether some clusters
need the large model more often.

Usage: python -m router.clustering
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from router.config import RESULTS_DIR, SEED
from router.features import load_features
from router.llm_summary import ci95
from router.plotting import GRID, INK, INK_2, MARKERS, MUTED, SERIES, SURFACE, apply_style, plt, save

PCA_DIM = 50
K_RANGE = range(2, 21)
MAX_K = 8  # one distinct color + marker pair per cluster


def main():
    _, emb, df = load_features()
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    Z = pca.fit_transform(emb)
    explained = pca.explained_variance_ratio_

    sil, assignments = {}, {}
    for k in K_RANGE:
        assignments[k] = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Z)
        sil[k] = float(silhouette_score(Z, assignments[k]))
        print(f"k={k} silhouette={sil[k]:.4f}")
    best_k = max((k for k in sil if k <= MAX_K), key=sil.get)
    df["cluster"] = assignments[best_k]

    clusters = summarize_clusters(df)
    table = pd.crosstab(df["cluster"], df["label"])
    chi2, p_value, dof, _ = chi2_contingency(table)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(RESULTS_DIR / "clusters.csv", index=False, float_format="%.4f")
    df[["qid", "cluster"]].to_csv(RESULTS_DIR / "cluster_assignments.csv", index=False)
    summary = {
        "pca_dim": PCA_DIM,
        "pca_explained_variance": float(explained.sum()),
        "pca_2d_explained_variance": float(explained[:2].sum()),
        "silhouette_by_k": sil,
        "max_k": MAX_K,
        "chosen_k": best_k,
        "silhouette": sil[best_k],
        "chi2_cluster_vs_label": {"chi2": chi2, "dof": dof, "p_value": p_value},
        "overall_large_label_rate": df["label"].mean(),
    }
    (RESULTS_DIR / "clustering_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(clusters.to_string(index=False))
    print(json.dumps(summary, indent=2))

    apply_style()
    plot_clusters(Z, df, sil, best_k, explained)
    plot_usage(clusters, df["label"].mean())


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in df.groupby("cluster"):
        top = g["subject"].value_counts().head(3)
        rows.append({
            "cluster": c,
            "n": len(g),
            "large_label_rate": g["label"].mean(),
            "large_label_ci95": ci95(g["label"].mean(), len(g)),
            "small_acc": g["small_correct"].mean(),
            "large_acc": g["large_correct"].mean(),
            "mean_prompt_tokens": g["n_tokens"].mean(),
            "top_category": g["category"].value_counts().idxmax(),
            "top_category_share": g["category"].value_counts(normalize=True).max(),
            "top_subjects": "; ".join(f"{s} ({n})" for s, n in top.items()),
        })
    return pd.DataFrame(rows)


def short_name(top_subjects: str, n: int = 2) -> str:
    names = [s.split(" (")[0].replace("high_school_", "HS ").replace("_", " ") for s in top_subjects.split("; ")]
    return ", ".join(names[:n])


def plot_clusters(Z, df, sil, best_k, explained):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})

    ks = list(sil)
    ax1.plot(ks, [sil[k] for k in ks], color=SERIES[0], marker="o", markersize=6,
             markeredgecolor=SURFACE, markeredgewidth=2)
    ax1.plot([best_k], [sil[best_k]], marker="o", markersize=11, color=SERIES[0],
             markeredgecolor=SURFACE, markeredgewidth=2)
    ax1.annotate(f"chosen k = {best_k}\nsilhouette = {sil[best_k]:.3f}", (best_k, sil[best_k]),
                 xytext=(10, -28), textcoords="offset points", fontsize=9, color=INK_2)
    ax1.axvspan(MAX_K + 0.5, ks[-1] + 0.5, color=GRID, alpha=0.5, linewidth=0)
    ax1.text(MAX_K + 0.8, min(sil.values()), "k > 8 not used:\nhard to interpret", fontsize=8, color=MUTED,
             va="bottom")
    ax1.set_xlim(ks[0] - 0.5, ks[-1] + 0.5)
    ax1.set_xticks([k for k in ks if k % 2 == 0])
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Silhouette score")
    ax1.set_title("a) K-Means silhouette score by k")

    # Scatter on the first two principal components; hue x marker keeps clusters apart under CVD.
    for c in sorted(df["cluster"].unique()):
        m = (df["cluster"] == c).to_numpy()
        ax2.scatter(Z[m, 0], Z[m, 1], s=9, color=SERIES[c], marker=MARKERS[c], alpha=0.45, linewidths=0,
                    label=f"Cluster {c}")
    for c in sorted(df["cluster"].unique()):
        m = (df["cluster"] == c).to_numpy()
        cx, cy = np.median(Z[m, 0]), np.median(Z[m, 1])
        ax2.text(cx, cy, str(c), ha="center", va="center", fontsize=10, fontweight="bold", color=INK,
                 bbox={"boxstyle": "circle,pad=0.25", "facecolor": SURFACE, "edgecolor": SERIES[c], "linewidth": 1.5})
    ax2.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    ax2.set_ylabel(f"PC2 ({explained[1]:.1%} of variance)")
    ax2.set_title("b) Prompt embeddings, PCA projection colored by K-Means cluster")
    ax2.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), markerscale=2.2, handletextpad=0.3)
    for h in ax2.get_legend().legend_handles:
        h.set_alpha(1)

    fig.tight_layout()
    save(fig, "fig3_pca_kmeans_clusters.png")


def plot_usage(clusters: pd.DataFrame, overall_rate: float):
    clusters = clusters.sort_values("large_label_rate").reset_index(drop=True)
    y = np.arange(len(clusters))
    labels = [f"Cluster {r.cluster}: {short_name(r.top_subjects)}" for r in clusters.itertuples()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 0.55 * len(clusters) + 1.8), sharey=True,
                                   gridspec_kw={"width_ratios": [1.2, 1]})
    colors = [SERIES[c] for c in clusters["cluster"]]
    ax1.barh(y, clusters["large_label_rate"], height=0.55, color=colors, edgecolor="none")
    ax1.errorbar(clusters["large_label_rate"], y, xerr=clusters["large_label_ci95"], fmt="none",
                 ecolor=INK_2, elinewidth=1)
    for yi, r in zip(y, clusters.itertuples()):
        ax1.text(r.large_label_rate + r.large_label_ci95 + 0.006, yi, f"{r.large_label_rate:.1%}  (n={r.n})",
                 va="center", fontsize=8.5, color=INK_2)
    ax1.axvline(overall_rate, color=MUTED, linewidth=1)
    ax1.set_ylim(-0.6, len(clusters) - 0.05)
    ax1.text(overall_rate, len(clusters) - 0.55, f" overall {overall_rate:.1%}", color=MUTED, fontsize=8,
             va="bottom")
    ax1.set_yticks(y, labels)
    ax1.set_xlim(0, (clusters["large_label_rate"] + clusters["large_label_ci95"]).max() * 1.35)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax1.set_xlabel("Questions that need the large model (label = LARGE)")
    ax1.set_title("a) Large-model need by cluster (95% CI)")
    ax1.grid(axis="y", visible=False)

    for col, color, label in [("small_acc", SERIES[0], "Small model"), ("large_acc", SERIES[1], "Large model")]:
        ax2.scatter(clusters[col], y, s=55, color=color, edgecolor=SURFACE, linewidth=2, zorder=3, label=label)
    for yi, r in zip(y, clusters.itertuples()):
        ax2.plot([r.small_acc, r.large_acc], [yi, yi], color=MUTED, linewidth=1, zorder=2)
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax2.set_xlabel("QA accuracy")
    ax2.set_title("b) Small vs large accuracy by cluster")
    ax2.grid(axis="y", visible=False)
    ax2.legend(loc="lower left")

    fig.tight_layout()
    save(fig, "fig4_large_model_usage_by_cluster.png")


if __name__ == "__main__":
    main()
