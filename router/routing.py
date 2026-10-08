"""Step 6: evaluate the full routing system on the held-out test split (Figure 5).

Policies compared:
  1. Always small     2. Always large     3. Random routing (same % sent to large as the router)
  4. Learned router, in two flavours:
       - prompt features only: decides before any model runs, so each question costs
         either the small OR the large model
       - + small-model confidence: a cascade that always runs the small model first, so a
         question sent to the large model costs small + large
  An oracle router (sends exactly the LARGE-labelled questions) is shown as an upper bound.

Each learned router is evaluated at two operating points, both chosen on the training split
(out-of-fold scores) and then applied unchanged to the test split:
  - budget:  send the training LARGE rate to the large model (`routing_threshold`)
  - quality: the cheapest threshold that keeps QUALITY_TARGET of the large model's accuracy
The full threshold sweep is plotted too. A model-gap table repeats the confidence-only router
with the 3B ("mid") model as the large model, to show how the size gap changes routing.

Compute is estimated as forward-pass FLOPs, 2 * n_params * prompt_tokens per question, and
reported relative to "always large". Measured CPU latency gives a second, empirical estimate.

Usage: python -m router.routing
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from router.config import LLM_META, LLM_OUTPUTS, RESULTS_DIR, SEED, SPLIT
from router.plotting import AXIS as GRID_STRONG, INK, INK_2, MUTED, SERIES, SURFACE, apply_style, plt, save

N_RANDOM_DRAWS = 1000
N_BOOTSTRAP = 2000
QUALITY_TARGET = 0.95
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


def curve_summary(curve: pd.DataFrame, acc_small: float, acc_large: float) -> dict:
    """RouteLLM-style summary [Ong et al. 2024] of an accuracy curve over the share of large-model calls.

    PGR (performance gap recovered) = (accuracy - small) / (large - small). APGR is the area under
    PGR from 0% to 100% of calls (random routing = 0.5); CPT(x%) is the smallest share of calls
    to the large model that recovers x% of the gap.
    """
    c = curve.sort_values(["pct_large", "accuracy"]).groupby("pct_large", as_index=False)["accuracy"].max()
    grid = np.linspace(0, 1, 1001)
    pgr = (np.interp(grid, c["pct_large"], c["accuracy"]) - acc_small) / (acc_large - acc_small)
    out = {"apgr": float(np.trapezoid(pgr, grid))}
    for level in (0.5, 0.8):
        ok = c[(c["accuracy"] - acc_small) / (acc_large - acc_small) >= level]
        out[f"cpt{int(level * 100)}"] = float(ok["pct_large"].min()) if len(ok) else np.nan
    return out


def quality_threshold(scores, small_correct, large_correct, cost_small, cost_large, cascade, target):
    """Cheapest threshold whose routing accuracy reaches `target` x always-large accuracy."""
    goal = target * large_correct.mean()
    best = (np.inf, np.inf)  # (cost, threshold); inf threshold = never escalate
    for t in np.unique(scores):
        st = policy_stats(scores >= t, small_correct, large_correct, cost_small, cost_large, cascade)
        if st["accuracy"] >= goal and st["cost"] < best[0]:
            best = (st["cost"], t)
    return best[1] if np.isfinite(best[0]) else -np.inf  # unreachable -> escalate everything


def model_gap_table(train, test, meta, rng) -> pd.DataFrame:
    """Confidence-only cascade router with each candidate large model (no training involved)."""
    rows = []
    for size in ("mid", "large"):
        tr_s, tr_l = train["small_correct"].to_numpy(), train[f"{size}_correct"].to_numpy()
        te_s, te_l = test["small_correct"].to_numpy(), test[f"{size}_correct"].to_numpy()
        tr_score, te_score = 1 - train["small_conf"].to_numpy(), 1 - test["small_conf"].to_numpy()
        tr_cost = [2 * meta[m]["n_params"] * train["n_tokens"].to_numpy() for m in ("small", size)]
        te_cost = [2 * meta[m]["n_params"] * test["n_tokens"].to_numpy() for m in ("small", size)]
        te_label = (1 - te_s) * te_l
        budget_t = np.quantile(tr_score, 1 - ((1 - tr_s) * tr_l).mean())
        quality_t = quality_threshold(tr_score, tr_s, tr_l, *tr_cost, True, QUALITY_TARGET)
        row = {"large_model": meta[size]["model"], "large_params_b": meta[size]["n_params"] / 1e9,
               "small_compute_share": meta["small"]["n_params"] / meta[size]["n_params"],
               "always_small_acc": te_s.mean(), "always_large_acc": te_l.mean(),
               "large_label_rate": te_label.mean(), "both_wrong_rate": ((1 - te_s) * (1 - te_l)).mean(),
               "conf_router_auc": roc_auc_score(te_label, te_score)}
        curve = pd.DataFrame([policy_stats(te_score >= t, te_s, te_l, *te_cost, cascade=True)
                              for t in np.unique(np.r_[np.inf, np.quantile(te_score, np.linspace(0, 1, 201))])])
        row |= {f"conf_router_{k}": v for k, v in curve_summary(curve, te_s.mean(), te_l.mean()).items()}
        for name, t in (("budget", budget_t), ("quality", quality_t)):
            route = te_score >= t
            st = policy_stats(route, te_s, te_l, *te_cost, cascade=True)
            gain, lo, hi = gain_vs_random(route, te_s, te_l, rng)
            row |= {f"{name}_accuracy": st["accuracy"], f"{name}_pct_large": st["pct_large"],
                    f"{name}_compute": st["cost"] / te_cost[1].sum(), f"{name}_gain_vs_random": gain,
                    f"{name}_gain_ci_low": lo, f"{name}_gain_ci_high": hi}
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(LLM_OUTPUTS).merge(pd.read_csv(SPLIT), on="qid")
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    preds = pd.read_csv(RESULTS_DIR / "router_test_predictions.csv")
    oof = pd.read_csv(RESULTS_DIR / "router_train_oof.csv")
    assert (preds["qid"].to_numpy() == test["qid"].to_numpy()).all()
    assert (oof["qid"].to_numpy() == train["qid"].to_numpy()).all()
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
    tr_s, tr_l = train["small_correct"].to_numpy(), train["large_correct"].to_numpy()
    tr_cost = [2 * meta[m]["n_params"] * train["n_tokens"].to_numpy() for m in ("small", "large")]
    for fs, cascade in ROUTERS.items():
        sel = metrics[(metrics["feature_set"] == fs) & metrics["selected"]].iloc[0]
        proba = preds[f"{sel['model']}|{fs}"].to_numpy()
        thresholds = {
            "budget": sel["routing_threshold"],
            "quality": quality_threshold(oof[f"{sel['model']}|{fs}"].to_numpy(), tr_s, tr_l, *tr_cost,
                                         cascade, QUALITY_TARGET),
        }
        for point, threshold in thresholds.items():
            route = proba >= threshold
            learned = {"policy": f"{ROUTER_LABELS[fs]} [{sel['model']}]", "operating_point": point,
                       "threshold": threshold, **evaluate(route, cascade)}
            # Random routing that sends the same share of questions to the large model.
            draws = [evaluate(rng.random(n) < route.mean()) for _ in range(N_RANDOM_DRAWS)]
            random_row = {"policy": f"Random routing (matched to {ROUTER_LABELS[fs].lower()})",
                          "operating_point": point, **{k: np.mean([d[k] for d in draws]) for k in draws[0]},
                          "accuracy_std": np.std([d["accuracy"] for d in draws])}
            learned["gain_vs_random"], learned["gain_ci_low"], learned["gain_ci_high"] = gain_vs_random(
                route, s_ok, l_ok, rng)
            rows += [random_row, learned]

        thresholds = np.unique(np.concatenate([[np.inf], np.quantile(proba, np.linspace(0, 1, 201))]))
        curves[fs] = pd.DataFrame([{"threshold": t, **evaluate(proba >= t, cascade)} for t in thresholds])
        curves[fs].to_csv(RESULTS_DIR / f"routing_curve_{fs.replace('+', '_')}.csv", index=False,
                          float_format="%.5f")

    conf_score = 1 - test["small_conf"].to_numpy()
    conf_thresholds = np.unique(np.concatenate([[np.inf], np.quantile(conf_score, np.linspace(0, 1, 201))]))
    curves_ref = {"confidence only": pd.DataFrame([evaluate(conf_score >= t, True) for t in conf_thresholds])}

    rows.append({"policy": "Oracle router (upper bound)", **evaluate(test["label"].to_numpy())})
    table = pd.DataFrame(rows)
    for kind in costs:
        table[f"compute_saved_{kind}"] = 1 - table[f"compute_{kind}"]
    table.to_csv(RESULTS_DIR / "routing_results.csv", index=False, float_format="%.4f")
    print(table.to_string(index=False))

    acc_small, acc_large = table.loc[0, "accuracy"], table.loc[1, "accuracy"]
    summary = [{"router": name, "large_model": meta["large"]["model"],
                **curve_summary(curve, acc_small, acc_large)}
               for name, curve in [*((ROUTER_LABELS[fs], c) for fs, c in curves.items()), *curves_ref.items()]]
    # The same metric on out-of-fold training predictions: a second estimate that does not use the test split.
    for fs, cascade in ROUTERS.items():
        sel = metrics[(metrics["feature_set"] == fs) & metrics["selected"]].iloc[0]
        score = oof[f"{sel['model']}|{fs}"].to_numpy()
        ts = np.unique(np.r_[np.inf, np.quantile(score, np.linspace(0, 1, 201))])
        train_curve = pd.DataFrame([policy_stats(score >= t, tr_s, tr_l, *tr_cost, cascade) for t in ts])
        summary.append({"router": ROUTER_LABELS[fs] + " (train CV)", "large_model": meta["large"]["model"],
                        **curve_summary(train_curve, tr_s.mean(), tr_l.mean())})
    summary.append({"router": "Random routing", "large_model": meta["large"]["model"],
                    "apgr": 0.5, "cpt50": 0.5, "cpt80": 0.8})
    v1 = RESULTS_DIR / "v1_large_3b"  # snapshot of the first iteration (3B as the large model)
    if v1.exists():
        v1_table = pd.read_csv(v1 / "routing_results.csv")
        for fs in ROUTERS:
            summary.append({"router": ROUTER_LABELS[fs] + " (v1)", "large_model": "Qwen/Qwen2.5-3B-Instruct",
                            **curve_summary(pd.read_csv(v1 / f"routing_curve_{fs.replace('+', '_')}.csv"),
                                            v1_table.loc[0, "accuracy"], v1_table.loc[1, "accuracy"])})
    summary = pd.DataFrame(summary)
    summary.to_csv(RESULTS_DIR / "routing_summary.csv", index=False, float_format="%.4f")
    print(summary.to_string(index=False))

    gap = model_gap_table(train, test, meta, rng)
    gap.to_csv(RESULTS_DIR / "model_gap.csv", index=False, float_format="%.4f")
    print(gap.T.to_string())

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
    from matplotlib.lines import Line2D

    apply_style()
    fig, ax = plt.subplots(figsize=(9, 5.4))
    small, large = table.iloc[0], table.iloc[1]
    oracle = table[table["policy"].str.startswith("Oracle")].iloc[0]

    ax.plot([small["compute_flops"], large["compute_flops"]], [small["accuracy"], large["accuracy"]],
            color=MUTED, linewidth=1.5, label="Random routing (expected)")
    goal = QUALITY_TARGET * large["accuracy"]
    ax.axhline(goal, color=GRID_STRONG, linewidth=1, zorder=1)
    ax.text(1.0, goal, f" {QUALITY_TARGET:.0%} of large-model accuracy", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8, color=MUTED)
    markers = {"budget": "o", "quality": "s"}
    for i, (fs, curve) in enumerate(curves.items()):
        curve = curve.sort_values("compute_flops")
        ax.plot(curve["compute_flops"], curve["accuracy"], color=SERIES[i], label=ROUTER_LABELS[fs])
        ops = table[table["policy"].str.startswith(ROUTER_LABELS[fs])]
        for _, op in ops.iterrows():
            ax.plot(op["compute_flops"], op["accuracy"], marker=markers[op["operating_point"]], markersize=9,
                    color=SERIES[i], markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", zorder=4)
            if fs != "prompt+conf":
                continue  # label the cascade's operating points only; the table carries the rest
            above = op["operating_point"] == "quality"
            ax.annotate(f"{op['accuracy']:.1%} acc, {op['pct_large']:.0%} to large,\n"
                        f"{1 - op['compute_flops']:.0%} compute saved",
                        (op["compute_flops"], op["accuracy"]), xytext=(-30, 34) if above else (28, -46),
                        textcoords="offset points", fontsize=8.5, color=INK_2, ha="right" if above else "left",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})

    for row, name, marker, offset in [(small, "Always small", "v", (8, -14)), (large, "Always large", "D", (-8, 8)),
                                      (oracle, "Oracle router", "*", (8, 4))]:
        ax.plot(row["compute_flops"], row["accuracy"], marker=marker, markersize=15 if marker == "*" else 8,
                color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=5)
        ax.annotate(f"{name}\n{row['accuracy']:.1%}", (row["compute_flops"], row["accuracy"]), xytext=offset,
                    textcoords="offset points", fontsize=8.5, color=INK,
                    ha="right" if offset[0] < 0 else "left")

    ax.set_ylim(small["accuracy"] - 0.03, oracle["accuracy"] + 0.025)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("Estimated compute (FLOPs, % of always using the large model)")
    ax.set_ylabel("Final QA accuracy (test split)")
    ax.set_title("Accuracy vs compute as the router threshold varies")
    handles, _ = ax.get_legend_handles_labels()
    handles += [Line2D([], [], marker=m, color=INK_2, linestyle="none", markersize=7,
                       label="budget operating point" if k == "budget" else f"{QUALITY_TARGET:.0%}-quality operating point")
                for k, m in markers.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    fig.tight_layout()
    save(fig, "fig5_accuracy_vs_compute.png")


if __name__ == "__main__":
    main()
