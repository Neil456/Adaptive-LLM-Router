"""Step 4: train and compare supervised routers (Figure 2).

Three classifiers (Logistic Regression, Random Forest, MLP) predict LARGE vs SMALL from two
feature sets:
  - "prompt":      prompt length, number/math-symbol counts, category, PCA of the embedding
  - "prompt+conf": the same plus the small model's confidence (requires running the small
                   model first, i.e. a cascade)

Hyperparameters are tuned with 5-fold stratified CV on the training split (ROC-AUC). Two
thresholds are picked from out-of-fold training predictions, so the test split is only
touched for the final numbers:
  - threshold:         maximizes F1; used for the classification metrics
  - routing_threshold: sends the same share of questions to the large model as the training
                       LARGE rate; used as the router's default operating point. (With a weak
                       classifier the F1-optimal threshold sends most questions to the large
                       model, which defeats the purpose of routing.)

Usage: python -m router.supervised
"""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from router.config import RESULTS_DIR, SEED
from router.features import CATEGORICAL_FEATURES, CONFIDENCE_FEATURES, NUMERIC_FEATURES, load_features
from router.plotting import INK_2, MUTED, SERIES, SURFACE, apply_style, bar, plt, save

EMB_PCA_DIM = 32
FEATURE_SETS = {"prompt": False, "prompt+conf": True}  # name -> uses small-model confidence
FEATURE_SET_LABELS = {"prompt": "Prompt features", "prompt+conf": "Prompt features + small-model confidence"}
MODELS = {
    "Logistic Regression": (
        LogisticRegression(class_weight="balanced", max_iter=5000),
        {"clf__C": [0.003, 0.01, 0.03, 0.1, 1.0]},
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=SEED),
        {"clf__max_depth": [6, 12, None], "clf__min_samples_leaf": [1, 5, 20]},
    ),
    "MLP": (
        MLPClassifier(max_iter=1000, early_stopping=True, random_state=SEED),
        {"clf__hidden_layer_sizes": [(32,), (64, 32)], "clf__alpha": [1e-3, 1e-2, 1e-1, 1.0]},
    ),
}
FEATURE_GROUPS = {
    "prompt length": ["prompt_tokens"],
    "numbers": ["num_numbers"],
    "math symbols": ["num_math_symbols"],
    "category": ["category"],
    "embedding": None,  # filled with the emb_* columns
    "small-model confidence": ["small_conf"],
}


def make_pipeline_for(model, use_conf: bool, emb_cols: list[str]) -> Pipeline:
    transformers = [
        # Counts are right-skewed, so log-transform before standardizing.
        ("num", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        # 384-d embedding -> 32 whitened principal components (unit variance, like the scaled features).
        ("emb", PCA(n_components=EMB_PCA_DIM, whiten=True, random_state=SEED), emb_cols),
    ]
    if use_conf:
        transformers.append(("conf", StandardScaler(), CONFIDENCE_FEATURES))
    return Pipeline([("prep", ColumnTransformer(transformers)), ("clf", model)])


def best_f1_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def base_rate_threshold(y: np.ndarray, proba: np.ndarray) -> float:
    """Threshold that flags the same fraction of questions as LARGE as the true label rate."""
    return float(np.quantile(proba, 1 - y.mean()))


def evaluate(y: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pct_pred_large": pred.mean(),
    }


def grouped_permutation_importance(pipe, X, y, groups, n_repeats=20, seed=SEED) -> dict:
    """Drop in test ROC-AUC when all columns of a feature group are shuffled together."""
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, pipe.predict_proba(X)[:, 1])
    out = {}
    for name, cols in groups.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - roc_auc_score(y, pipe.predict_proba(Xp)[:, 1]))
        out[name] = (float(np.mean(drops)), float(np.std(drops)))
    return out


def difficulty_diagnostic(X_train, df_train, emb_cols, cv) -> pd.DataFrame:
    """CV ROC-AUC of the prompt-feature logistic regression for three targets.

    Separates "is this question hard?" (either model wrong) from "does escalating help?"
    (the LARGE label), which is what a router actually needs.
    """
    targets = {
        "small model wrong": 1 - df_train["small_correct"].to_numpy(),
        "large model wrong": 1 - df_train["large_correct"].to_numpy(),
        "LARGE label (small wrong, large right)": df_train["label"].to_numpy(),
    }
    rows = []
    for name, y in targets.items():
        model = LogisticRegression(class_weight="balanced", max_iter=5000, C=0.01)
        scores = cross_val_score(make_pipeline_for(model, False, emb_cols), X_train, y, cv=cv, scoring="roc_auc")
        rows.append({"target": name, "positive_rate": y.mean(), "cv_roc_auc": scores.mean(),
                     "cv_roc_auc_std": scores.std()})
    return pd.DataFrame(rows)


def main():
    feats, emb, df = load_features()
    emb_cols = [f"emb_{i}" for i in range(emb.shape[1])]
    X = pd.concat([feats.drop(columns=["qid", "split"]), pd.DataFrame(emb, columns=emb_cols)], axis=1)
    y = df["label"].to_numpy()
    train, test = (df["split"] == "train").to_numpy(), (df["split"] == "test").to_numpy()
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"train={train.sum()} test={test.sum()} LARGE rate train={y_train.mean():.3f} test={y_test.mean():.3f}")

    rows = []
    preds = pd.DataFrame({"qid": df.loc[test, "qid"].to_numpy(), "label": y_test})
    fitted = {}

    # Reference rows: the majority class, and thresholding the small model's confidence alone.
    rows.append({"feature_set": "-", "model": "Majority class (always SMALL)", "cv_roc_auc": 0.5,
                 "cv_roc_auc_std": 0.0, "threshold": np.nan, "best_params": "",
                 **evaluate(y_test, np.zeros(len(y_test)), 0.5)})
    conf_train, conf_test = 1 - X_train["small_conf"].to_numpy(), 1 - X_test["small_conf"].to_numpy()
    t = best_f1_threshold(y_train, conf_train)
    rows.append({"feature_set": "conf only", "model": "Small-model confidence threshold",
                 "cv_roc_auc": roc_auc_score(y_train, conf_train), "cv_roc_auc_std": 0.0, "threshold": t,
                 "routing_threshold": base_rate_threshold(y_train, conf_train),
                 "best_params": "", **evaluate(y_test, conf_test, t)})
    preds["Confidence threshold|conf only"] = conf_test

    for fs_name, use_conf in FEATURE_SETS.items():
        for model_name, (model, grid) in MODELS.items():
            pipe = make_pipeline_for(model, use_conf, emb_cols)
            search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)
            oof = cross_val_predict(clone(search.best_estimator_), X_train, y_train, cv=cv,
                                    method="predict_proba", n_jobs=-1)[:, 1]
            threshold = best_f1_threshold(y_train, oof)
            proba = search.predict_proba(X_test)[:, 1]
            row = {
                "feature_set": fs_name,
                "model": model_name,
                "cv_roc_auc": search.best_score_,
                "cv_roc_auc_std": search.cv_results_["std_test_score"][search.best_index_],
                "threshold": threshold,
                "routing_threshold": base_rate_threshold(y_train, oof),
                "best_params": json.dumps({k.removeprefix("clf__"): v for k, v in search.best_params_.items()}),
                **evaluate(y_test, proba, threshold),
            }
            rows.append(row)
            preds[f"{model_name}|{fs_name}"] = proba
            fitted[(fs_name, model_name)] = search.best_estimator_
            print(f"{fs_name:12s} {model_name:20s} cv_auc={row['cv_roc_auc']:.3f} "
                  f"test acc={row['accuracy']:.3f} f1={row['f1']:.3f} auc={row['roc_auc']:.3f}")

    metrics = pd.DataFrame(rows)
    # The router used downstream is chosen by CV ROC-AUC on the training split, never by test score.
    metrics["selected"] = False
    for fs_name in FEATURE_SETS:
        cands = metrics[metrics["feature_set"] == fs_name]
        metrics.loc[cands["cv_roc_auc"].idxmax(), "selected"] = True
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "supervised_metrics.csv", index=False, float_format="%.4f")
    preds.to_csv(RESULTS_DIR / "router_test_predictions.csv", index=False, float_format="%.6f")
    print(metrics.drop(columns=["best_params"]).to_string(index=False))

    groups = {k: (emb_cols if v is None else v) for k, v in FEATURE_GROUPS.items()}
    importance = []
    for _, sel in metrics[metrics["selected"]].iterrows():
        pipe = fitted[(sel["feature_set"], sel["model"])]
        for group, (mean, std) in grouped_permutation_importance(pipe, X_test, y_test, groups).items():
            importance.append({"feature_set": sel["feature_set"], "model": sel["model"], "feature_group": group,
                               "auc_drop_mean": mean, "auc_drop_std": std})
    importance = pd.DataFrame(importance)
    importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False, float_format="%.4f")
    print(importance.to_string(index=False))

    diagnostic = difficulty_diagnostic(X_train, df[train], emb_cols, cv)
    diagnostic.to_csv(RESULTS_DIR / "difficulty_diagnostic.csv", index=False, float_format="%.4f")
    print(diagnostic.to_string(index=False))

    plot(metrics)


def plot(metrics: pd.DataFrame):
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    models = list(MODELS)
    x = np.arange(len(models))
    w = 0.34
    majority = metrics[metrics["model"].str.startswith("Majority")].iloc[0]
    conf_only = metrics[metrics["feature_set"] == "conf only"].iloc[0]
    for ax, metric, title in zip(axes, ["accuracy", "f1", "roc_auc"], ["Accuracy", "F1 (LARGE class)", "ROC-AUC"]):
        for i, (fs_name, offset) in enumerate([("prompt", -w / 2), ("prompt+conf", w / 2)]):
            vals = metrics.set_index(["feature_set", "model"]).loc[fs_name].loc[models, metric]
            bar(ax, x + offset, vals, SERIES[i], w, label=FEATURE_SET_LABELS[fs_name])
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        # Reference lines, labelled in an empty strip to the right of the bars.
        refs = {"accuracy": [(majority["accuracy"], "always\nSMALL")],
                "f1": [(conf_only["f1"], "confidence\nonly")],
                "roc_auc": [(conf_only["roc_auc"], "confidence\nonly"), (0.5, "chance")]}
        for ref_val, ref_label in refs[metric]:
            ax.axhline(ref_val, color=MUTED, linewidth=1)
            ax.text(len(models) - 0.6, ref_val, ref_label, ha="left", va="center", fontsize=8, color=MUTED,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
        ax.set_xlim(-0.55, len(models) + 0.1)
        ax.set_xticks(x, ["Logistic\nRegression", "Random\nForest", "MLP"])
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Held-out test score")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig2_supervised_comparison.png")


if __name__ == "__main__":
    main()
