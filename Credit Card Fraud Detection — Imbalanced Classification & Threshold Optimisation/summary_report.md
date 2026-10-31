# Summary Report: Credit Card Fraud Detection

> **Demo run:** these numbers come from a synthetic stand-in dataset. Re-run the notebook with the real `creditcard.csv` to regenerate this report.

## 1. Business problem and why accuracy fails
A digital payments company processes millions of transactions a day, and only about 0.17% are fraudulent. Each missed fraud costs roughly Rs.4,500, while reviewing a flagged transaction costs about Rs.150 of analyst time. A model that labels every transaction "legitimate" scores 99.84% accuracy yet has 0% recall and stops no fraud at all. Accuracy therefore hides failure on the class that costs money, so models here are judged on Precision, Recall, F1 and above all PR-AUC, which ignores the sea of easy true negatives.

## 2. Imbalance handling
Three training sets were prepared from the training split only: the original data, SMOTE (fraud raised to 10% of the legitimate count) and random undersampling (also 10%). Logistic Regression performed best on the **SMOTE** version by PR-AUC, so Random Forest used that version with balanced class weights. XGBoost kept the original data and used `scale_pos_weight` = 644.2, its built-in way to up-weight fraud. The test set was never resampled, so every score reflects the real class mix.

## 3. Model and threshold recommended
The **tuned XGBoost** (randomized search, 15 candidates, 3-fold CV on average precision; best CV PR-AUC 0.782) reaches a test PR-AUC of 0.726 (Random Forest 0.381, Logistic Regression 0.862). Note: Logistic Regression scored the highest PR-AUC (0.862) in this run, so confirm the choice before deploying. At the F1-optimal threshold of **0.8846** it delivers Precision 1.000, Recall 0.562 and F1 0.720. If the risk team demands at least 90% recall, a threshold of 0.0001 gives Recall 0.938 at Precision 0.065. The default 0.5 cut-off is not used, because it ignores the cost asymmetry between a missed fraud and a false alarm.

## 4. Cost-benefit result
On the 10,000-transaction test set, the F1-optimal threshold catches 9 frauds, misses 7 and raises 0 false alarms. That saves Rs.40,500, costs Rs.1,350 in investigations and yields a **net benefit of Rs.39,150**, which is about **Rs.195,750 per 50,000 transactions**. The best net benefit among the thresholds tested was Rs.43,050 (threshold 0.3000); the F1-optimal threshold earns less, which shows that ML metrics and business metrics can diverge.

## 5. What I would improve in production
* **Real-time scoring:** serve the pipeline behind a low-latency API with a feature store for velocity features (transactions per card per hour) and a fast fallback rule if the service is down.
* **Drift monitoring:** track input drift (Amount, Hour, V-features via PSI) and rolling PR-AUC, precision and recall once labels arrive, and alert when they degrade.
* **Feedback loop:** feed analyst decisions and chargebacks back as labels, retrain on a schedule, and pick thresholds on a validation window rather than the test set.
* **Business tuning:** re-estimate the Rs.4,500 fraud value and Rs.150 review cost regularly, and consider tiers (auto-block, step-up authentication, manual review) instead of one cut-off.
