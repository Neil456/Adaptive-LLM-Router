"""Step 6: evaluate the full routing system on the held-out test split (Figure 5).

Policies compared:
  1. Always small     2. Always large     3. Random routing (same % sent to large as the router)
  4. Learned router, in two flavours:
       - prompt features only: decides before any model runs, so each question costs
         either the small OR the large model
       - + small-model confidence: a cascade that always runs the small model first, so a
         question sent to the large model costs small + large
  An oracle router (sends exactly the LARGE-labelled questions) is shown as an upper bound.

The learned routers' default operating point is `routing_threshold` from the supervised step
(sends the training LARGE rate to the large model); the full threshold sweep is plotted too.

Compute is estimated as forward-pass FLOPs, 2 * n_params * prompt_tokens per question, and
reported relative to "always large". Measured CPU latency gives a second, empirical estimate.

Usage: python -m router.routing
"""
import json

import numpy as np
import pandas as pd

from router.config import LLM_META, LLM_OUTPUTS, RESULTS_DIR, SEED, SPLIT
from router.plotting import INK, INK_2, MUTED, SERIES, SURFACE, apply_style, plt, save

N_RANDOM_DRAWS = 1000
N_BOOTSTRAP = 2000
ROUTERS = {"prompt": False, "prompt+conf": True}  # feature set -> is a cascade (small model always runs)
ROUTER_LABELS = {"prompt": "Learned router (prompt features)",
                 "prompt+conf": "Learned router (+ small-model confidence)"}


def policy_stats(route_large, small_correct, large_correct, cost_small, cost_large, cascade=False) -> dict:
    """Accuracy, share routed to large and cost (summed per question) for a boolean routing vector."""
    route_large = np.asarray(route_large, dtype=bool)
    correct = np.where(route_large, large_correct, small_correct)
    if cascade:
        cost = cost_small.sum() + cost_large[route_large].sum()
    else:
        cost = np.where(route_large, cost_large, cost_small).sum()
    return {"accuracy": correct.mean(), "pct_large": route_large.mean(), "cost": cost}


def gain_vs_random(route, small_correct, large_correct, rng, n_boot=N_BOOTSTRAP):
    """Router accuracy minus the expected accuracy of random routing at the same share, with a
    95% paired-bootstrap CI over test questions."""
    route = np.asarray(route, dtype=bool)
    p = route.mean()

    def gain(idx):
        s, l, r = small_correct[idx], large_correct[idx], route[idx]
        return np.where(r, l, s).mean() - ((1 - p) * s.mean() + p * l.mean())

    n = len(route)
    boots = [gain(rng.integers(0, n, n)) for _ in range(n_boot)]
    return gain(np.arange(n)), *np.percentile(boots, [2.5, 97.5])


def main():
    df = pd.read_csv(LLM_OUTPUTS).merge(pd.read_csv(SPLIT), on="qid")
    test = df[df["split"] == "test"].reset_index(drop=True)
    preds = pd.read_csv(RESULTS_DIR / "router_test_predictions.csv")
    assert (preds["qid"].to_numpy() == test["qid"].to_numpy()).all()
    metrics = pd.read_csv(RESULTS_DIR / "supervised_metrics.csv")
    meta = json.loads(LLM_META.read_text())

    s_ok, l_ok = test["small_correct"].to_numpy(), test["large_correct"].to_numpy()
    costs = {
        # FLOPs of one forward pass over the prompt
        "flops": (2 * meta["small"]["n_params"] * test["n_tokens"].to_numpy(),
                  2 * meta["large"]["n_params"] * test["n_tokens"].to_numpy()),
        # measured single-question CPU latency (seconds)
        "latency": (test["small_latency_s"].to_numpy(), test["large_latency_s"].to_numpy()),
    }

    def evaluate(route, cascade=False):
        out = {}
        for kind, (cs, cl) in costs.items():
            st = policy_stats(route, s_ok, l_ok, cs, cl, cascade)
            out["accuracy"], out["pct_large"] = st["accuracy"], st["pct_large"]
            out[f"compute_{kind}"] = st["cost"] / cl.sum()  # relative to always-large
        return out

    rng = np.random.default_rng(SEED)
    n = len(test)
    rows = [
        {"policy": "Always small", **evaluate(np.zeros(n))},
        {"policy": "Always large", **evaluate(np.ones(n))},
    ]
    curves = {}
    for fs, cascade in ROUTERS.items():
        sel = metrics[(metrics["feature_set"] == fs) & metrics["selected"]].iloc[0]
        proba = preds[f"{sel['model']}|{fs}"].to_numpy()
        route = proba >= sel["routing_threshold"]
        learned = {"policy": f"{ROUTER_LABELS[fs]} [{sel['model']}]", "threshold": sel["routing_threshold"],
                   **evaluate(route, cascade)}

        # Random routing that sends the same share of questions to the large model.
        draws = [evaluate(rng.random(n) < route.mean()) for _ in range(N_RANDOM_DRAWS)]
        random_row = {"policy": f"Random routing (matched to {ROUTER_LABELS[fs].lower()})",
                      **{k: np.mean([d[k] for d in draws]) for k in draws[0]},
                      "accuracy_std": np.std([d["accuracy"] for d in draws])}
        learned["gain_vs_random"], learned["gain_ci_low"], learned["gain_ci_high"] = gain_vs_random(
            route, s_ok, l_ok, rng)
        rows += [random_row, learned]

        thresholds = np.unique(np.concatenate([[np.inf], np.quantile(proba, np.linspace(0, 1, 201))]))
        curves[fs] = pd.DataFrame([{"threshold": t, **evaluate(proba >= t, cascade)} for t in thresholds])
        curves[fs].to_csv(RESULTS_DIR / f"routing_curve_{fs.replace('+', '_')}.csv", index=False,
                          float_format="%.5f")

    rows.append({"policy": "Oracle router (upper bound)", **evaluate(test["label"].to_numpy())})
    table = pd.DataFrame(rows)
    for kind in costs:
        table[f"compute_saved_{kind}"] = 1 - table[f"compute_{kind}"]
    table.to_csv(RESULTS_DIR / "routing_results.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False))

    # Cheapest point on each curve that recovers >= X% of the accuracy gain from small -> large.
    # Random routing recovers X% of the gain (in expectation) by sending X% of questions to large.
    acc_small, acc_large = table.loc[0, "accuracy"], table.loc[1, "accuracy"]
    small_compute = table.loc[0, "compute_flops"]
    budget = {}
    for frac in (0.5, 0.8, 0.9):
        target = acc_small + frac * (acc_large - acc_small)
        entry = {"target_accuracy": target,
                 "random": {"compute_flops": small_compute + frac * (1 - small_compute), "pct_large": frac}}
        for fs, curve in curves.items():
            ok = curve[curve["accuracy"] >= target]
            best = None if ok.empty else ok.loc[ok["compute_flops"].idxmin()]
            entry[fs] = None if best is None else {"compute_flops": float(best["compute_flops"]),
                                                   "pct_large": float(best["pct_large"])}
        budget[f"{int(frac * 100)}%_of_gain"] = entry
    (RESULTS_DIR / "routing_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    print(json.dumps(budget, indent=2))

    plot(table, curves)


def plot(table: pd.DataFrame, curves: dict):
    apply_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    small, large = table.iloc[0], table.iloc[1]
    oracle = table[table["policy"].str.startswith("Oracle")].iloc[0]

    ax.plot([small["compute_flops"], large["compute_flops"]], [small["accuracy"], large["accuracy"]],
            color=MUTED, linewidth=1.5, label="Random routing (expected)")
    for i, (fs, curve) in enumerate(curves.items()):
        curve = curve.sort_values("compute_flops")
        ax.plot(curve["compute_flops"], curve["accuracy"], color=SERIES[i], label=ROUTER_LABELS[fs])
        op = table[table["policy"].str.startswith(ROUTER_LABELS[fs])].iloc[0]
        ax.plot(op["compute_flops"], op["accuracy"], marker="o", markersize=9, color=SERIES[i],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
        ax.annotate(f"{op['accuracy']:.1%} acc, {op['pct_large']:.0%} to large,\n"
                    f"{1 - op['compute_flops']:.0%} compute saved",
                    (op["compute_flops"], op["accuracy"]), xytext=[(25, -55), (-30, 40)][i],
                    textcoords="offset points", fontsize=8.5, color=INK_2, ha=["left", "right"][i],
                    arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})

    for row, name, marker, offset in [(small, "Always small", "s", (8, -14)), (large, "Always large", "D", (-8, 8)),
                                      (oracle, "Oracle router", "*", (8, 4))]:
        ax.plot(row["compute_flops"], row["accuracy"], marker=marker, markersize=15 if marker == "*" else 8,
                color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=5)
        ax.annotate(f"{name}\n{row['accuracy']:.1%}", (row["compute_flops"], row["accuracy"]), xytext=offset,
                    textcoords="offset points", fontsize=8.5, color=INK,
                    ha="right" if offset[0] < 0 else "left")

    ax.set_ylim(small["accuracy"] - 0.025, oracle["accuracy"] + 0.025)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Estimated compute (FLOPs, % of always using the large model)")
    ax.set_ylabel("Final QA accuracy (test split)")
    ax.set_title("Accuracy vs compute as the router threshold varies")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "fig5_accuracy_vs_compute.png")


if __name__ == "__main__":
    main()
