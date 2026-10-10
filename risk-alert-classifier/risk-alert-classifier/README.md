# Risk Alert Classifier

Binary classification project that predicts whether a bank customer is **High Risk (1)** or **Low Risk (0)** from
demographic, transaction-activity and credit-behaviour data, with a full study of class imbalance, tree-based models,
hyperparameter tuning and ROC-based model selection.

> **Business goal:** catch as many truly high-risk customers as possible. A missed high-risk customer (false negative, Type-II) is much more
> expensive than an unnecessary review (false positive, Type-I), so the final model is chosen to minimise false negatives while keeping precision workable.

## Headline result

| | |
|---|---|
| **Final model** | Tuned Logistic Regression (L1, `C=0.3`, `class_weight='balanced'`, threshold 0.50) |
| Test set (920 customers, 111 high risk) | **0 false negatives, 2 false positives**; recall 1.000, precision 0.982, F1 0.991, AUC 1.000 |
| Repeated 5x2 cross-validation | recall 0.988 +/- 0.012, precision 0.961, AUC 0.9995 |
| Why not accuracy? | Always predicting "Low Risk" scores 87.9% accuracy with 0% recall |

The near-perfect scores come with a caution: no feature leaks the label, but the data behaves as if it were synthetic and almost linearly separable.
Expect lower performance on real data (see *Limitations* in the final report).

## Repository structure

```
risk-alert-classifier/
|-- Risk_Alert_Classifier.ipynb     # main notebook: Parts A-H, executed, with outputs and interpretations
|-- data/
|   `-- risk_alert_classifier_dataset.csv
|-- figures/                         # every chart from the notebook (PNG screenshots)
|-- screenshots/                     # copies of the key figures used in this README
|-- results/                         # leaderboard, CV comparison, final metrics (CSV)
|-- models/risk_alert_model.joblib   # final model + preprocessing + threshold
|-- docs/
|   |-- Theory_Concepts.pdf          # definitions of all theory concepts (Part A and more)
|   `-- Final_Report.pdf             # Part H report
|-- src/
|   |-- nb_cells.py, interp.py, run_notebook.py   # builds/executes the notebook
|   |-- score_new_customers.py   # CLI scoring of raw customer rows
|   `-- make_theory_pdf.py, make_final_report.py, pdf_common.py
|-- requirements.txt
`-- README.md
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate      # optional
pip install -r requirements.txt
jupyter notebook Risk_Alert_Classifier.ipynb            # Run All
```

The notebook runs top to bottom in roughly 6-8 minutes on one CPU (most of the time is the Random Forest searches).
To rebuild the notebook and PDFs from the sources: `python src/run_notebook.py && PYTHONPATH=src python src/make_final_report.py`.

## Task coverage

| Part | Tasks | What the notebook contains |
|---|---|---|
| A | 1-6 | Short theory answers (Logistic Regression, metrics, Type-I/II, AUC-ROC, imbalance); longer definitions in `docs/Theory_Concepts.pdf` |
| B | 7-9 | Feature/target identification, leakage check, stratified 80/20 split, missing-value analysis, KNN imputation fitted on train only |
| C | 10-12 | Logistic Regression baseline, confusion matrix, metrics, Type-I/Type-II errors |
| D | 13-15 | Impact of imbalance, under-sampling, over-sampling, SMOTE, ADASYN, before/after comparison (recall, F1, AUC); extended tree-model and severe-imbalance experiments |
| E | 16-19 | Decision Tree, overfitting analysis (depth curve), Random Forest, comparison |
| F | 20-22 | Randomized Search CV (DT, RF), Grid Search CV (RF, LR), tuned vs untuned |
| G | 23-25 | ROC curves, AUC table, repeated-CV model selection with a precision floor, threshold analysis |
| H | 26-27 | Final report in the notebook and as PDF, cost illustration, saved model and scoring function |

## Key findings

1. **Imbalance (12.1% High Risk) makes accuracy useless.** The majority-class dummy gets 87.9% accuracy and 0% recall.
2. **Logistic Regression is the best model on this data.** The risk label is close to a weighted combination of about nine credit-behaviour features,
   so linear models are near-perfect while a single tree overfits (train F1 1.000 vs test 0.869).
3. **Resampling helps where the model is weak.** Decision Tree recall: 0.838 -> 0.93. Under severe imbalance (1.8%) Random Forest recall falls to 0.793;
   SMOTE/ADASYN restore it to 0.856 and under-sampling to 1.000 (at 13 false positives). Plain over-sampling made it worse (0.712). Logistic Regression gained nothing (already 1.000).
4. **Tuning with an F2 score** trades a few false positives for fewer misses, in line with the business requirement.
5. **Top risk drivers:** late-payment days, missed payments, credit utilisation, debt balance, low income and credit score.

## Screenshots

| | |
|---|---|
| ![Target and missing](screenshots/01_target_and_missing.png) | ![Baseline confusion matrix](screenshots/04_baseline_confusion_matrix.png) |
| ![Balancing LR](screenshots/06_balancing_comparison.png) | ![Balancing trees](screenshots/06b_balancing_trees.png) |
| ![DT overfitting](screenshots/08_dt_overfitting_curve.png) | ![ROC](screenshots/10_roc_all_models.png) |
| ![Threshold](screenshots/12_threshold_analysis.png) | ![Final CM](screenshots/13_final_confusion_matrix.png) |

## Scoring new customers

```bash
python src/score_new_customers.py path/to/customers.csv scored.csv
```

The script loads `models/risk_alert_model.joblib` (model, scaler, KNN imputer, threshold) and applies the identical
feature engineering, encoding and imputation to raw rows with the original CSV columns. It returns
`risk_probability` and `high_risk_alert` (1 = send for review).

## Methodology notes

* **No leakage:** the split is done first; scaler, KNN imputer and resamplers are fitted on training data only; the test set is used once for reporting.
* **Resampling backend:** the notebook uses `imbalanced-learn` when installed, otherwise a built-in implementation of the same algorithms. The committed outputs
  were generated with the built-in version because the library was not available in the build environment; re-running with the library may change numbers slightly.
* **Selection rule:** highest repeated-CV recall among models with CV precision >= 0.95, then threshold check on out-of-fold predictions.
