# Spam Message Classification: KNN vs SVM vs Naive Bayes

Classify messages as **spam (1)** or **legitimate (0)** from engineered message, sender and time features, and compare three classifier families:

| Family | Model |
|---|---|
| Distance-based | K-Nearest Neighbors (KNN) |
| Margin-based | Support Vector Machine (linear, RBF, polynomial kernels) |
| Probabilistic | Gaussian Naive Bayes (with a hand-written Bayes' theorem implementation) |

## Repository layout

```
.
├── data/message_intelligence_dataset.csv   # 5,200 messages, 16 columns
├── notebook/spam_classification.ipynb      # full analysis: Parts A-G (executed, with outputs)
├── figures/                                # 16 labelled charts + final_results.csv, operating_points.csv
├── report/Spam_Classification_Report.pdf   # theory definitions, results, conclusions
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
cd notebook
jupyter notebook spam_classification.ipynb      # or: jupyter nbconvert --to notebook --execute --inplace spam_classification.ipynb
```

The notebook uses relative paths, so run it from the `notebook/` folder. A fixed random seed (42) makes every result reproducible; it runs in about a minute.

## Dataset

5,200 rows. Target: `spam_label` (973 spam = 18.7%, 4,227 legitimate).

| Group | Columns |
|---|---|
| Message | `message_length`, `word_count`, `num_urls`, `num_digits`, `num_special_chars` |
| Keywords | `spam_keyword_score`, `legit_keyword_score` |
| Sender | `sender_activity_score`, `sender_account_age_days`, `messages_sent_last_24h` |
| Time | `hour_of_day`, `day_of_week` (`timestamp` is redundant with these) |
| Not used | `message_id` (identifier), `message_text` (raw text; its signal is summarised by the numeric features) |

About 2-3% of values are missing in `sender_activity_score`, `sender_account_age_days` and `messages_sent_last_24h`.

## Method

1. **Split:** 80/20 stratified train/test (4,160 / 1,040), seed 42.
2. **Preprocessing (fit on train only, inside sklearn Pipelines):** median imputation, then `StandardScaler` for KNN and SVM. Naive Bayes uses imputed raw values (scale-invariant, interpretable parameters).
3. **Model selection:** 5-fold stratified cross-validation on the training set only (K and metric for KNN; kernel, C, gamma, degree for SVM). The test set is used once for the final comparison.
4. **Evaluation:** accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrices, ROC/PR curves, threshold analysis, 5-fold CV on all data, permutation importance and a label-noise robustness experiment.

## Results (held-out test set, 1,040 messages, 195 spam)

| Model | Accuracy | Precision | Recall | F1 | False alarms | Missed spam |
|---|---|---|---|---|---|---|
| KNN (K=1, Euclidean, scaled) | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| SVM (linear, C=0.01) | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 0 |
| Gaussian Naive Bayes | 0.999 | 0.995 | 1.000 | 0.997 | 1 | 0 |

Robustness to label noise (test F1 when a share of training labels is flipped):

| Model | 0% | 5% | 10% | 20% |
|---|---|---|---|---|
| KNN (K=1) | 1.000 | 0.875 | 0.767 | 0.601 |
| KNN (K=15) | 0.997 | 0.997 | 0.997 | 0.991 |
| SVM (linear) | 1.000 | 1.000 | 1.000 | 1.000 |
| Naive Bayes | 0.997 | 0.988 | 0.961 | 0.935 |

## Key findings

* **The dataset is almost trivially separable.** `num_special_chars > 0` alone classifies all 5,200 messages correctly (every spam message has one, no legitimate message does). The near-perfect scores of every model therefore show the pipeline works, not that real-world spam detection is this easy. Model differences show up in robustness, not accuracy.
* **Scaling matters for KNN:** without it, account age dominates the distance and CV F1 falls from 1.00 to about 0.90 (Euclidean). The misclassified messages in that case are spam from older accounts and legitimate messages from very new accounts.
* **SVM:** the problem is linearly separable (margin stops changing from C = 1; only 10 support vectors). The linear kernel matches RBF and polynomial and is the most robust to noisy labels.
* **Naive Bayes:** the hand-computed Bayes posterior matches scikit-learn to ~1e-16. Its independence assumption is violated (e.g. message length vs URLs r = 0.77), making its probabilities over-confident (99.9% of test messages get P < 0.01 or > 0.99). Its single error is a threshold issue: its ROC-AUC is still 1.0.
* **Recommendation:** deploy the linear SVM as the main filter, keep Naive Bayes as an explainable fast secondary signal (with calibration or a raised threshold), and use KNN only with K around 15 and scaling. Re-validate on real, noisy, adversarial data before production, and monitor the dependence on `num_special_chars`.

## Assignment coverage

| Part | Where |
|---|---|
| A: theory (Q1-5) | Notebook "Part A" and the PDF report |
| B: features, preprocessing, split (Q6-8) | Notebook "Part B" |
| C: KNN, K sweep, metrics, errors (Q9-12) | Notebook "Part C" (errors analysed on stress-test configurations, since tuned KNN makes none) |
| D: SVM, margins, support vectors, vs KNN (Q13-15) | Notebook "Part D" |
| E: Naive Bayes, manual probabilities, Bayes' theorem (Q16-19) | Notebook "Part E" |
| F: metrics, comparison, best precision / recall (Q20-22) | Notebook "Part F" |
| G: final report and recommendation (Q23-24) | Notebook "Part G" and `report/Spam_Classification_Report.pdf` |

## Publishing to GitHub

```bash
git init
git add .
git commit -m "Spam message classification: KNN vs SVM vs Naive Bayes"
git branch -M main
git remote add origin https://github.com/<your-username>/spam-message-classifier.git
git push -u origin main
```

