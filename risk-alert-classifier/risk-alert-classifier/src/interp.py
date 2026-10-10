I = {}

I["eda1"] = """
**Interpretation.** The target is imbalanced: 4,043 Low-Risk customers (87.9%) vs 557 High-Risk (12.1%), roughly 7.3 : 1.
Seven columns have missing values, between 2.2% (`region`) and 4.7% (`credit_score`) of rows, so no column is badly affected and
imputation (rather than dropping columns or rows) is appropriate.
"""

I["eda2"] = """
**Interpretation.** High-risk customers clearly differ on the credit-behaviour features: they have more late-payment days
(mean 16.3 vs 4.1), more missed payments (3.3 vs 0.6), higher credit utilisation (0.67 vs 0.36), more cash advances and failed logins,
and lower credit scores (589 vs 690) and income. Features such as `age`, `account_tenure_months` and `monthly_transaction_count`
look the same for both classes and are expected to carry little signal.
"""

I["leakage"] = """
**Interpretation.** No single feature gives the answer away: the strongest, `avg_late_payment_days`, has a single-feature AUC of 0.953,
and several features sit near 0.5 (no signal alone). `customer_id` was removed up front. Risk is therefore determined by the
*combination* of about nine credit-behaviour variables, which is exactly what a multivariate model should learn. This also explains
the very high scores seen later (see the caution in Part C).
"""

I["imputation"] = """
**Interpretation.** Imputation preserves the shape of each distribution: e.g. mean credit score 677.6 -> 677.7 and mean income
41,924 -> 41,831, with a slight narrowing of the spread (about 1-2%), which is the normal smoothing effect of averaging neighbours.
All 7 columns have 0 missing values afterwards in both train and test, giving a 27-column numeric feature matrix.
"""

I["baseline"] = """
**Interpretation.** On the 920-row test set the baseline Logistic Regression produced **809 TN, 0 FP, 0 FN, 111 TP**, i.e. 1.000 for
accuracy, precision, recall, F1 and AUC.

* **Type-I errors (false positives): 0**. No low-risk customer was wrongly flagged.
* **Type-II errors (false negatives): 0**. No high-risk customer was missed.

**Caution.** Perfect scores are unusual on real data. The leakage check found no feature that reveals the label, so the most likely
explanation is that this dataset is cleanly (almost linearly) separable, which is typical of synthetic data. The test set holds only
111 positives, so the honest expectation is a small error rate rather than exactly zero. Cross-validation in Part G gives a recall of
about 0.975 for this same model. The largest drivers are income (negative) and late-payment days, utilisation, debt balance
and missed payments (positive).
"""

I["imbalance_demo"] = """
**Interpretation.** A "model" that ignores every feature and always answers *Low Risk* reaches **87.9% accuracy** yet has
**0% recall**: it misses all 111 high-risk customers. Accuracy rewards the majority class, so with a 7 : 1 imbalance it cannot be
trusted. This is the core problem of imbalanced data: the metric that matters (recall on the rare class) is invisible in accuracy.
The Logistic Regression baseline avoids the trap only because the classes happen to be strongly separable here.
"""

I["balancing"] = """
**Interpretation (Logistic Regression).** The baseline is already at recall 1.000 and AUC 1.000, so balancing has nothing to gain, and
every technique keeps recall at 1.000 while adding one or two false positives (precision 0.982-0.991 instead of 1.000):

| Technique | FP added | F1 |
|---|---|---|
| Over-sampling | 1 | 0.996 |
| SMOTE | 1 | 0.996 |
| Under-sampling | 2 | 0.991 |
| ADASYN | 2 | 0.991 |

Resampling moves the decision boundary toward the majority class, which catches more minority cases but also flags a few more
low-risk customers. When the classes are already well separated there is no minority recall to recover. The differences (1-2
customers out of 809) are within noise. Balancing matters when the model struggles with the minority class, which the next
experiment checks.
"""

I["balancing_trees"] = """
**Interpretation.**

* **Decision Tree** benefits clearly. Recall rises from 0.838 (18 missed) to 0.928-0.937 (7-8 missed) and AUC from 0.913 to 0.949-0.958.
  Over-sampling gives the best balance (F1 0.920 vs 0.869). Under-sampling reaches high recall but at a heavy price in
  false positives (46, precision 0.69).
* **Random Forest** is already strong (recall 0.982, 2 missed, 0 false positives). Under-sampling lifts recall to 1.000 (0 missed) but adds 11
  false positives; SMOTE keeps recall at 0.982 with one false positive; AUC stays at about 0.9999 throughout.

Balancing trades precision for recall. It helps most for the weaker, high-variance learner (the single tree).
"""

I["stress"] = """
**Interpretation.** With only 60 high-risk examples (1.8%) the damage of imbalance becomes visible for the flexible model: the
Random Forest's recall collapses from 0.982 to **0.793 (23 missed customers)**, while Logistic Regression holds at 0.991, because
its simple linear boundary needs few positives to be estimated.

Remedies on the Random Forest: **over-sampling does not help** (recall 0.712, the duplicated points are simply memorised),
**SMOTE/ADASYN recover part of it** (0.856), and **under-sampling restores recall to 1.000** but at the cost of 13 false positives.
Under-sampling also lifts Logistic Regression to recall 1.000 at 17 false positives (precision 0.867). So for severe imbalance,
rebalancing (or class weights) is essential for tree ensembles, and every method trades precision for recall.
"""

I["dt_overfit"] = """
**Interpretation.** The fully grown tree (depth 17, 74 leaves) memorises the training data (accuracy, recall and F1 all **1.000**)
but drops to recall 0.838, F1 0.869 and AUC 0.913 on the test set: an F1 gap of **0.131**, the signature of overfitting. The depth
curve shows the training F1 climbing to 1.0 while the test F1 peaks around depth 8 (about 0.90) and then stalls or falls. Limiting depth
or pruning is therefore needed, which is exactly what the hyperparameter search in Part F does.
"""

I["dt_vs_rf"] = """
**Interpretation.** Both models reach 1.000 on the training data, but they generalise very differently:

| | Decision Tree | Random Forest |
|---|---|---|
| Test accuracy | 0.970 | 0.998 |
| Test recall | 0.838 (18 missed) | 0.982 (2 missed) |
| Test F1 | 0.869 | 0.991 |
| Test AUC | 0.913 | 0.9999 |
| Train-test F1 gap | 0.131 | 0.009 |

Averaging 100 de-correlated trees removes most of the variance of a single tree. The Random Forest's most important features are
`avg_late_payment_days` (0.27), `missed_payments_12m` (0.20), `credit_score` (0.12), `credit_utilization_ratio` (0.11) and
`cash_advance_count_6m` (0.09), consistent with the exploratory analysis. Note that the simple Logistic Regression baseline still matches
or beats the Random Forest on this data.
"""

I["tuning"] = """
**Interpretation (tuning scored with F2, so recall counts more).**

* **Decision Tree:** the search chose a shallow, regularised tree (`max_depth=4`, entropy, `class_weight='balanced'`, `ccp_alpha=0.001`).
  Recall improves from 0.838 to 0.973 (missed customers 18 -> 3) and AUC from 0.913 to 0.983, but precision falls to 0.783
  (30 false positives), so F1 is unchanged. It trades false positives for fewer misses.
* **Random Forest:** the tuned forest (`max_features='log2'`, `min_samples_leaf=8`, `class_weight='balanced_subsample'`) has recall 1.000
  (0 missed, was 2) with 3 false positives (was 0). **Grid Search** refined the result only marginally (CV F2 0.9650 -> 0.9655) and the
  test confusion matrix is identical: the Randomized Search had already found the plateau.
* **Logistic Regression:** the best setting is `C=0.3`, L1 penalty and `class_weight='balanced'` (CV F2 0.983). On the test set it
  has 0 missed customers and 2 false positives.

Tuned models are better for the *business goal* (fewer false negatives), not for accuracy or precision.
"""

I["roc"] = """
**Interpretation.** Every model except the untuned Decision Tree (AUC 0.913) has an AUC of at least 0.983, and the Logistic Regression variants
and Random Forests are essentially perfect (0.9999-1.000). The curves hug the top-left corner, and the zoomed panel is needed to see
differences: the default tree is clearly lower, the tuned tree is slightly lower, and the others overlap.

Because AUC cannot separate the top models (the test set has only 111 positives), the final choice is made with recall, precision
and cross-validation at the actual operating point, not AUC alone.
"""

I["selection"] = """
**Interpretation.** Repeated CV ranks **Tuned Logistic Regression (Grid)** first: recall **0.988 +/- 0.012**, precision 0.961, F2 0.982,
AUC 0.9995. It beats the tuned Random Forests (recall 0.964-0.966) and the baseline Logistic Regression (0.975). Both Decision
Trees are excluded by the precision floor (0.885 and 0.755), and the default Random Forest has the lowest recall of the eligible models (0.931). On
the test set the selected model has **0 false negatives and 2 false positives**.

A simple linear model wins because the risk label here is close to a weighted combination of the credit-behaviour features, so extra
model flexibility adds variance without adding signal. It is also the most explainable option, which matters for risk decisions.
"""

I["final_model"] = """
**Interpretation.** For the class-weighted Logistic Regression the F2-optimal threshold from out-of-fold predictions is **0.50**:
`class_weight='balanced'` has already shifted the probabilities toward the high-risk class. A lower cut-off (0.32, out-of-fold
recall >= 0.99) is available as a high-recall option. On the test set it caught no additional customers (0 missed at both thresholds) but doubled
the false positives (2 -> 4), so **0.50 is kept**. Lower the threshold only if the business accepts more reviews in exchange for a
safety margin on recall.
"""

I["final_report"] = """
---
## Final report

### 1. Best model and justification
**Tuned Logistic Regression** (`C=0.3`, L1 penalty, `class_weight='balanced'`, decision threshold 0.50, standardised inputs).

| Evidence | Result |
|---|---|
| Test set (920 customers, 111 high risk) | **0 false negatives, 2 false positives**, recall 1.000, precision 0.982, F1 0.991, AUC 1.000 |
| Repeated 5x2 CV (training set) | recall 0.988 +/- 0.012, precision 0.961, F2 0.982, AUC 0.9995 (best eligible model) |
| Business requirement | fewest missed high-risk customers at an acceptable precision |
| Other advantages | explainable coefficients, fast, stable, no resampling needed |

### 2. Impact of the imbalance-handling techniques
* The data is 87.9% / 12.1%. Accuracy is misleading: always predicting "Low Risk" scores 87.9% with zero recall.
* **Logistic Regression**: balancing did not improve recall (already 1.000) and added 1-2 false positives. For this model `class_weight` inside the
  tuned model is sufficient.
* **Decision Tree**: balancing raised recall from 0.838 to 0.93-0.94 and AUC from 0.913 to 0.95-0.96. Over-sampling was the best compromise (F1 0.920).
* **Random Forest**: under-sampling reached recall 1.000 (11 false positives); SMOTE kept recall at 0.982 with 1 false positive.
* **Severe imbalance (1.8%)**: Random Forest recall fell to 0.793; SMOTE/ADASYN recovered it to 0.856 and under-sampling to 1.000 (13 false positives). Plain
  over-sampling made it worse (0.712).
* Every technique trades precision for recall; none is free.

### 3. Comparison of performance metrics (test set)
| Model | Recall | Precision | F1 | AUC | FN | FP |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline) | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| **Tuned Logistic Regression (Grid)** | **1.000** | **0.982** | **0.991** | **1.000** | **0** | **2** |
| Tuned Random Forest (Grid) | 1.000 | 0.974 | 0.987 | 1.000 | 0 | 3 |
| Random Forest (default) | 0.982 | 1.000 | 0.991 | 1.000 | 2 | 0 |
| Tuned Decision Tree | 0.973 | 0.783 | 0.868 | 0.983 | 3 | 30 |
| Decision Tree (default) | 0.838 | 0.903 | 0.869 | 0.913 | 18 | 10 |

### 4. Business interpretation of false positives and false negatives
* **False negative (Type-II):** a high-risk customer is cleared. This is the expensive error (undetected default/fraud exposure), so the project optimises for it first.
  The final model missed none on the test set; cross-validation suggests a realistic miss rate of roughly 1-2.5%.
* **False positive (Type-I):** a low-risk customer is flagged. Cost is analyst time and some customer friction. The final model produced 2 in 809 low-risk customers (0.25%) on
  the test set and about 4% of alerts were not truly high risk in cross-validation (precision 0.961). This workload is manageable.
* Under an assumed 10 : 1 cost ratio (FN : FP) the final model has one of the lowest expected costs (see the cost table above).

### 5. Conclusions and recommendations
1. Deploy the tuned Logistic Regression with threshold 0.50 as the alerting model, with human review of every alert.
2. Keep a **high-recall option (threshold about 0.32)** ready if the business wants a larger safety margin at the cost of more reviews.
3. Use class weights or under-sampling/SMOTE whenever a tree ensemble is used, and always report recall and precision rather than accuracy.
4. Monitor the model: track recall on confirmed outcomes, drift in `avg_late_payment_days`, utilisation and income, and retrain on a regular schedule.
5. Strongest risk signals to act on: late-payment days, missed payments, utilisation, debt balance, low income / low credit score.

### 6. Limitations
* The near-perfect scores, the absence of any leakage, and the almost linear structure suggest the dataset is **synthetic or unusually clean**. Real-world performance will probably be lower.
* The test set contains only 111 high-risk customers, so one error moves recall by about 0.9 points; that is why selection also used cross-validation.
* Imputation and scaling were fitted on the training set, but the CV folds inside it share one imputer fit (a very small optimistic effect).
* No fairness audit was done (e.g. the `Other` gender group has only 43 customers). One should be completed before real use.
* The resampling comparisons used the built-in implementations of SMOTE/ADASYN because `imbalanced-learn` was unavailable when the notebook outputs were generated.
  Install `imbalanced-learn` (see `requirements.txt`) and the notebook will use the library versions automatically; small numeric differences are possible.
"""
