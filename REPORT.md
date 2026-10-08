# Adaptive LLM Router: Predicting When a Small Model Is Enough

**Summary.** We ran 5,700 MMLU questions through a 0.5B and a 3B Qwen2.5 model, labelled each
question `LARGE` when only the large model answered correctly, and trained routers to predict
that label. Prompt features alone (length, category, number and symbol counts, sentence
embedding) carry almost no signal: the best classifier reaches a test ROC-AUC of 0.58, and
cross-validated AUC is 0.53. Adding the small model's own confidence raises test ROC-AUC to
0.66. Used as a cascade, that router sends 28% of questions to the large model, reaches
**56.3% accuracy** (always-small: 47.6%, always-large: 67.5%) and saves **58% of the compute**
of always using the large model. That is **+3.2 points over random routing** at the same
budget (95% CI +1.5 to +4.9). Clustering the embeddings shows that some topics need the large
model more often than others (22.7% to 30.8% of questions; χ² p = 0.003), but the spread is
small. The main obstacle is that question difficulty cuts both ways: hard questions defeat
the small model, but they often defeat the large model too.

---

## 1. Introduction

Larger language models are more accurate, but each answer also costs more. Many questions do
not need the big model: a small model answers them correctly at a fraction of the cost. An
**adaptive router** looks at each incoming question and decides which model should answer it.
If it can spot the questions the small model will get wrong *and* the large model will get
right, it can approach large-model accuracy while paying mostly small-model prices.

This project builds a deliberately simple version of that idea:

1. Run every question in a multiple-choice QA benchmark through a small and a large LLM.
2. Label each question `SMALL` or `LARGE` according to which model is actually needed.
3. Train three classifiers (Logistic Regression, Random Forest, MLP) to predict that label
   from cheap features of the prompt.
4. Cluster the prompt embeddings (PCA + K-Means) to see whether some topics need the large
   model more often than others.
5. Use the best classifier as a router and compare it with always-small, always-large and
   random routing on accuracy and estimated compute.

## 2. Dataset

**Questions.** MMLU (Hendrycks et al., 2021) contains four-option multiple-choice questions
in 57 subjects. We sample **100 test questions from every subject** (seed 42), giving **5,700
questions** with a mean prompt length of 103 tokens (range 33 to 990). Each subject is mapped
to one of MMLU's four super-categories (STEM, Humanities, Social Sciences, Other), which
serves as the question-category feature.

**Models.** Both models come from the same family, so they share a tokenizer and see
identical token sequences:

| Role  | Model                        | Parameters | Mean CPU latency / question |
|-------|------------------------------|-----------:|----------------------------:|
| Small | `Qwen/Qwen2.5-0.5B-Instruct` | 0.49 B     | 0.13 s |
| Large | `Qwen/Qwen2.5-3B-Instruct`   | 3.09 B     | 0.62 s |

**Answering protocol.** Zero-shot prompt in the lm-evaluation-harness style:

```
The following is a multiple choice question about {subject}.

{question}
A. {choice A}
B. {choice B}
C. {choice C}
D. {choice D}
Answer:
```

Each model runs one forward pass, and its answer is the letter whose token (` A` … ` D`) has
the highest next-token logit. The softmax over those four logits gives a probability per
option, and the probability of the chosen option is the model's **confidence**. Both models
ran in bfloat16 on a 4-core CPU.

**Labels.**

- `LARGE` (1): the small model is wrong **and** the large model is right
- `SMALL` (0): everything else

"Everything else" covers three cases: both models right, only the small model right, and both
wrong. When both are wrong, escalating costs more and fixes nothing, so the cheap model is the
right call.

![Figure 1](figures/fig1_small_vs_large_accuracy.png)

**Figure 1.** (a) Accuracy of each model by MMLU category, with 95% confidence intervals.
(b) How the two models' outcomes combine; the orange bar is the `LARGE` class.

| Category | n | Small acc. | Large acc. | `LARGE` rate |
|---|---:|---:|---:|---:|
| STEM | 1,800 | 40.4% | 59.1% | 30.2% |
| Humanities | 1,300 | 50.5% | 68.6% | 25.8% |
| Social Sciences | 1,200 | 55.3% | 76.6% | 27.9% |
| Other | 1,400 | 50.0% | 68.1% | 25.9% |
| **All** | **5,700** | **48.2%** | **67.2%** | **27.6%** |

The large model is 19 points more accurate overall and better in every category. Per
question, the outcomes break down as follows:

- **Both correct (39.5%) and only-small-correct (8.7%):** the small model already gets these
  right.
- **Only-large-correct (27.6%):** the `LARGE` class, the questions a router should escalate.
- **Both wrong (24.2%):** escalating these costs more and gains nothing.

The `LARGE` rate ranges from 14% (marketing) to 40% (high-school microeconomics) across
subjects (`results/accuracy_by_subject.csv`).

**Split.** 80/20 train/test split, stratified on the label (4,560 / 1,140 questions). All
model selection uses only the training split, through 5-fold cross-validation. The test split
is used once, for the final numbers.

## 3. Methods

### 3.1 Features and preprocessing

| Feature | Description | Preprocessing |
|---|---|---|
| Prompt length | tokens in the full prompt | log(1 + x), then standardization |
| Numbers | count of numbers in question + choices | log(1 + x), then standardization |
| Math symbols | count of `+ * / = ^ < > % √ π ∑ ∫ ≤ ≥ …` and minus signs | log(1 + x), then standardization |
| Category | MMLU super-category | one-hot encoding |
| Prompt embedding | `all-MiniLM-L6-v2` sentence embedding (384-d) of question + choices | PCA to 32 whitened components |
| *(optional)* Small-model confidence | probability of the small model's chosen answer | standardization |

All preprocessing lives inside a scikit-learn `Pipeline`/`ColumnTransformer`. Scalers, the
encoder and the PCA are therefore re-fitted inside every cross-validation fold, which keeps
test information from leaking into the transforms.

There are two feature sets:

- **Prompt features**: everything except confidence. The router can decide before any LLM
  runs.
- **Prompt features + small-model confidence**: needs the small model to run first. This
  makes the router a *cascade*: the small model always answers, and the router decides whether
  to escalate. Escalated questions pay for both models.

### 3.2 Supervised models

| Model | Settings | Hyperparameter grid (5-fold CV, ROC-AUC) |
|---|---|---|
| Logistic Regression | `class_weight="balanced"` | C ∈ {0.003, 0.01, 0.03, 0.1, 1} |
| Random Forest | 400 trees, `class_weight="balanced_subsample"` | max_depth ∈ {6, 12, None} × min_samples_leaf ∈ {1, 5, 20} |
| MLP | Adam, early stopping | hidden layers ∈ {(32), (64, 32)} × L2 alpha ∈ {0.001, 0.01, 0.1, 1} |

Each model reports **accuracy**, **F1 for the `LARGE` class** and **ROC-AUC** on the test
split. Accuracy and F1 need a hard decision, so the threshold on P(`LARGE`) is set to maximize
F1 on out-of-fold predictions from the training split. Two reference rows are included: the
majority class (always predict `SMALL`), and thresholding the small model's confidence alone.

### 3.3 Unsupervised analysis

The 384-d embeddings of all 5,700 questions are reduced to 50 dimensions with PCA, which
retains 49% of the variance. K-Means (10 restarts) is run for k = 2 … 20, with the silhouette
score computed for each k. Labels are not used at any point in clustering. They are joined
afterwards to compare, cluster by cluster, how often the large model is needed. A χ² test of
independence checks whether the `LARGE` rate differs between clusters.

### 3.4 Routing evaluation

All routing numbers are on the 1,140-question test split.

| Policy | Rule |
|---|---|
| Always small | every question goes to the small model |
| Always large | every question goes to the large model |
| Random routing | each question goes to the large model with probability *p*, where *p* matches the learned router's share (mean of 1,000 draws) |
| Learned router | the classifier with the best CV ROC-AUC sends a question to the large model when P(`LARGE`) ≥ threshold |
| Oracle (reference) | sends exactly the `LARGE`-labelled questions to the large model |

**Router threshold.** The F1-optimal threshold suits classification, but it is a poor routing
default. A weak classifier maximizes F1 by flagging almost every question as `LARGE` (Section
4), which sends nearly everything to the expensive model. The router's default operating
point therefore sends **the same share of questions as the training `LARGE` rate** (27.7%):
the threshold is the matching quantile of the out-of-fold training scores. Figure 5 also
sweeps the full threshold range, so every other trade-off is visible.

**Compute.** A forward pass over *T* prompt tokens costs about 2 · N<sub>params</sub> · *T*
FLOPs. Policy cost is summed over questions and reported relative to always-large. One small
pass costs 0.494 / 3.086 = **16%** of one large pass. For the cascade router, escalated
questions pay small + large. As an empirical check, we also report the measured CPU latency of
each forward pass under the same accounting.

## 4. Supervised results

| Features | Model | CV ROC-AUC (train) | Test accuracy | Test F1 | Test ROC-AUC |
|---|---|---:|---:|---:|---:|
| – | Majority class (always `SMALL`) | 0.500 | 0.724 | 0.000 | 0.500 |
| confidence only | Small-model confidence threshold | 0.619 | 0.504 | 0.495 | 0.638 |
| prompt | Logistic Regression | 0.521 ± 0.017 | 0.277 | 0.431 | 0.567 |
| prompt | **Random Forest** | **0.532 ± 0.017** | 0.284 | 0.436 | **0.579** |
| prompt | MLP | 0.522 ± 0.008 | 0.276 | 0.433 | 0.551 |
| prompt + conf | Logistic Regression | 0.612 ± 0.006 | **0.551** | 0.482 | 0.643 |
| prompt + conf | **Random Forest** | **0.625 ± 0.013** | 0.489 | **0.487** | **0.660** |
| prompt + conf | MLP | 0.593 ± 0.028 | 0.373 | 0.431 | 0.547 |

Bold models are the ones selected by CV ROC-AUC and used as routers. Full results, including
the selected hyperparameters, are in `results/supervised_metrics.csv`.

![Figure 2](figures/fig2_supervised_comparison.png)

**Figure 2.** Test-set accuracy, F1 and ROC-AUC for the three classifiers on both feature
sets. Gray lines are reference points: always predicting `SMALL` (accuracy), thresholding
small-model confidence alone (F1, ROC-AUC), and chance (ROC-AUC).

**Prompt features alone barely beat chance.** All three models have a cross-validated ROC-AUC
of 0.52 to 0.53 and a test ROC-AUC of 0.55 to 0.58. At their F1-optimal thresholds they flag
99 to 100% of questions as `LARGE`. Their F1 (≈ 0.43) and accuracy (≈ 0.28) are exactly what
"always predict `LARGE`" would score, since 2 · 0.276 / 1.276 = 0.433. Accuracy is a misleading
metric here: the majority class scores 0.72 while being useless as a router.

**Small-model confidence is the strongest single signal.** Thresholding it alone gives a test
ROC-AUC of 0.64. Combined with the prompt features, Random Forest reaches 0.66, and Logistic
Regression 0.64.

**Model choice matters less than features.** Random Forest is best on both feature sets, but
the differences between classifiers (≤ 0.03 AUC, except the MLP with confidence) are
comparable to their CV standard deviations. The MLP is the weakest and the least stable. With
only 4,560 training examples and a weak signal, it overfits: its CV AUC of 0.593 drops to 0.547
on test.

**What the Random Forest relies on.** We measured grouped permutation importance: the drop in
test ROC-AUC when one feature group is shuffled.

| Feature group | AUC drop: prompt router | AUC drop: + confidence router |
|---|---:|---:|
| Small-model confidence | – | **0.121** |
| Embedding | **0.076** | 0.032 |
| Numbers | 0.002 | 0.008 |
| Category | 0.001 | 0.000 |
| Prompt length, math symbols | ≤ 0 | ≤ 0 |

The embedding carries what little signal the prompt features have. Length, category and
symbol counts contribute essentially nothing once the embedding is present.

**Why the prompt features fail.** The same prompt-feature logistic regression was trained on
three different targets (5-fold CV on the training split, `results/difficulty_diagnostic.csv`):

| Target | Positive rate | CV ROC-AUC |
|---|---:|---:|
| Small model is wrong | 51.7% | 0.626 |
| Large model is wrong | 32.9% | 0.651 |
| `LARGE` label (small wrong **and** large right) | 27.6% | 0.517 |

The features *do* recognise hard questions: they predict each model's errors reasonably well.
But a router needs something narrower, the questions that are hard for the small model yet
still easy enough for the large one. Difficulty pushes that label in both directions. A harder
question is more likely to beat the small model, which raises the chance of `LARGE`. It is
also more likely to beat the large model, which lowers it. The two effects largely cancel.

## 5. Unsupervised results

![Figure 3](figures/fig3_pca_kmeans_clusters.png)

**Figure 3.** (a) Silhouette score of K-Means for k = 2 … 20. (b) Prompt embeddings projected
on the first two principal components, colored (and shaped) by cluster. Numbers mark each
cluster's median position.

**Cluster structure is weak.** The silhouette score is low for every k (0.043 to 0.080). It
rises slowly with k and never peaks, which means MMLU prompts form a continuum of topics rather
than well-separated groups. We use **k = 8** (silhouette 0.065), the best value up to 8. Beyond
that, gains are small (0.080 at k = 20) and the clusters become hard to interpret and display.
The 2-D projection in Figure 3b captures only 6% of embedding variance, so overlaps there
overstate the overlap in 50-D. The one cleanly separated cluster (7) is moral scenarios, whose
questions all share a fixed template.

**The clusters are topical.** Although no labels were used, each cluster aligns with a group
of subjects:

| Cluster | Main subjects | n | Small acc. | Large acc. | `LARGE` rate |
|---:|---|---:|---:|---:|---:|
| 4 | HS micro-/macroeconomics, econometrics | 770 | 44.8% | 68.6% | **30.8%** |
| 3 | HS & elementary mathematics, college physics | 892 | 31.4% | 48.0% | 30.0% |
| 7 | moral scenarios | 100 | 24.0% | 31.0% | 30.0% |
| 1 | anatomy, college biology, astronomy | 1,106 | 49.0% | 70.1% | 29.8% |
| 0 | formal logic, logical fallacies, professional law | 749 | 46.6% | 64.2% | 27.0% |
| 5 | HS world / European / US history | 845 | 59.9% | 79.8% | 25.3% |
| 6 | professional medicine, medical genetics, virology | 542 | 50.2% | 68.5% | 25.3% |
| 2 | management, marketing, public relations | 696 | 61.6% | 77.7% | **22.7%** |

![Figure 4](figures/fig4_large_model_usage_by_cluster.png)

**Figure 4.** (a) Share of each cluster's questions that need the large model, with 95%
confidence intervals; the vertical line is the overall rate. (b) Small and large model
accuracy in each cluster.

**Some clusters need the large model more often.** The cluster `LARGE` rate ranges from 22.7%
to 30.8%. A χ² test rejects independence between cluster and label (χ² = 21.8, 7 d.f.,
p = 0.003). The effect, though, is modest: an 8-point spread around a 27.6% average.

Figure 4b shows the same tension as the supervised diagnostic:

- **Easy clusters** (business, history) need the large model least, because the small model
  already does well there (60 to 62%).
- **Hard clusters** (mathematics, moral scenarios) do *not* need it much more, despite the
  small model scoring only 24 to 31%. The large model also struggles there (31 to 48%), so
  many questions end up "both wrong".
- **Mid-difficulty clusters** (economics, biology/astronomy) need the large model most. There
  the small model fails often (45 to 49%) but the large model usually succeeds (69 to 70%).

## 6. Routing results

| Policy (test split, n = 1,140) | QA accuracy | Sent to large | Compute (FLOPs) | Compute saved | Measured CPU time saved |
|---|---:|---:|---:|---:|---:|
| Always small | 47.6% | 0% | 16.0% | 84.0% | 78.8% |
| Always large | 67.5% | 100% | 100% | 0% | 0% |
| Random routing (p = 0.286) | 53.3% ± 0.8 | 28.6% | 40.0% | 60.0% | 56.2% |
| **Learned router, prompt features** (RF) | **54.8%** | 28.6% | 40.2% | **59.8%** | 55.1% |
| Random routing (p = 0.277) | 53.2% ± 0.8 | 27.8% | 39.3% | 60.7% | 57.0% |
| **Learned router, + small-model confidence** (RF, cascade) | **56.3%** | 27.7% | 42.3% | **57.7%** | 52.2% |
| Oracle router (upper bound) | 75.3% | 27.6% | 38.2% | 61.8% | 57.1% |

Random-routing accuracy is the mean ± standard deviation over 1,000 draws. Compute is
relative to always-large.

| Learned router | Gain over random at the same share | 95% bootstrap CI |
|---|---:|---:|
| Prompt features | +1.5 points | −0.2 to +3.1 |
| + small-model confidence (cascade) | **+3.2 points** | **+1.5 to +4.9** |

![Figure 5](figures/fig5_accuracy_vs_compute.png)

**Figure 5.** Test accuracy against estimated compute as the router threshold sweeps from
"send nothing" to "send everything". Large dots mark the default operating points. The gray
line is the expected accuracy of random routing. The cascade's curve extends past 100%
because escalated questions pay for both models.

**The learned router beats random routing, but not by much.** At the default operating point,
the cascade router reaches 56.3% accuracy while saving 58% of the compute. That is 3.2 points
better than random routing with the same budget, and the bootstrap CI excludes zero. The
prompt-only router gains 1.5 points, but its CI includes zero.

Either router keeps only a modest part of what routing could achieve:

- At ~28% of questions escalated, the routers recover **36% (prompt) to 44% (cascade)** of the
  20-point gap between always-small and always-large.
- An oracle with the same budget would reach **75.3%**. That is more than always-large,
  because in 8.7% of questions only the small model is right, and the oracle keeps those with
  the small model.

**Accuracy vs compute.** Figure 5 shows both router curves lying above the random-routing
line across most of the budget range. Read off the test curves (and so slightly optimistic,
since the threshold is picked on the test set), compute needed to recover a given share of the
small-to-large accuracy gain is:

| Share of the accuracy gain | Random routing | Prompt-feature router | Cascade router |
|---|---:|---:|---:|
| 50% (57.6% accuracy) | 58% | 49% | 50% |
| 80% (63.6% accuracy) | 83% | 74% | 79% |
| 90% (65.6% accuracy) | 92% | 82% | 92% |

Below about 45% compute the cascade is ahead by roughly a point, because confidence
identifies the small-model mistakes most worth fixing. Above about 70% the prompt-only router
wins. The cascade pays the 16% small-model overhead on every question, while the prompt-only
router does not.

**Measured vs estimated compute.** Measured CPU time agrees with the FLOP estimate in
direction but shows slightly smaller savings. On this CPU the small model is only 4.7× faster
than the large one, versus 6.25× fewer FLOPs, because fixed per-call overheads weigh more on
small models.

## 7. Limitations

- **Noisy labels.** On a four-option test, a model at chance still answers 25% correctly. Some
  `LARGE` labels are therefore lucky guesses, especially in near-chance clusters like moral
  scenarios (small 24%, large 31%). The labels come from a single deterministic run, so this
  noise cannot be averaged out.
- **One benchmark, one model pair, one prompt format.** MMLU, zero-shot, letter-logit scoring,
  and two models from the same family. Real deployments generate free-form answers, where
  output tokens dominate cost. Confidence would also have to come from sequence likelihoods or
  self-reports rather than four option logits. A wider gap between models (e.g. 0.5B vs 70B)
  would make routing both more valuable and possibly easier.
- **Simple compute model.** FLOPs count only the prompt forward pass. They ignore the router's
  own cost (the MiniLM encoder is small, about 0.7% of the large model's parameters) and
  memory and serving effects. Latency was measured at batch size 1 on a shared 4-core CPU.
- **Small test set.** 1,140 test questions give confidence intervals of about ±3 points on
  accuracy and about ±0.04 on ROC-AUC. Differences between the three classifiers are within
  this noise. The compute-budget table is read from the test curves and is mildly optimistic.
- **Weak cluster structure.** Silhouette scores below 0.1 mean the clusters partition a
  continuum rather than discover natural groups. The choice of k = 8 is partly a presentation
  choice.
- **Simple features by design.** Richer signals, such as the small model's hidden states,
  answer entropy across prompts, or a fine-tuned encoder, were out of scope.

## 8. Conclusion

We built a complete small-vs-large routing pipeline on 5,700 MMLU questions:

- **Supervised:** Logistic Regression, Random Forest and MLP routers, with full preprocessing
  (standardization, one-hot encoding, sentence embeddings, PCA).
- **Unsupervised:** PCA + K-Means analysis of the prompt embeddings.
- **Routing:** an end-to-end evaluation against always-small, always-large and random
  routing.

The headline result is honest but modest:

- **Prompt features alone hardly predict when the large model is needed** (cross-validated
  ROC-AUC 0.53).
- **The small model's confidence lifts this to 0.66.** As a cascade, the router then reaches
  56.3% accuracy at 42% of the large model's compute, 3.2 points better than random routing
  with the same budget.
- **Clustering confirms that large-model need varies by topic** (22.7% to 30.8%), in a
  pattern a single "difficulty" score cannot capture. Mid-difficulty topics such as economics
  benefit most from escalation, while the hardest topics often defeat both models.

The diagnostic in Section 4 points to what a better router needs. Easy-vs-hard detection
already works reasonably well: AUC 0.63 for small-model errors and 0.65 for large-model
errors. What a router needs on top is to separate *"hard for the small model"* from *"hard
for every model"*. Predicting the two error events separately is one natural next step;
signals taken from inside the small model are another.

---

*Reproduce with `SKIP_LLMS=1 ./run_pipeline.sh` (uses the committed LLM outputs) or
`./run_pipeline.sh` (re-runs both LLMs, ~1 hour on CPU). Every number above is in `results/`.*
