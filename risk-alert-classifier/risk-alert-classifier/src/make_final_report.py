"""Builds docs/Final_Report.pdf from the saved result tables and figures (Part H, task 26)."""
import pandas as pd
from PIL import Image as PILImage
from pdf_common import *


def img(name, width=17 * cm):
    path = f"figures/{name}.png"
    w, h = PILImage.open(path).size
    return Image(path, width=width, height=width * h / w)


lb = pd.read_csv("results/test_set_leaderboard.csv").set_index("Model")
cvt = pd.read_csv("results/cv_comparison.csv").set_index("Model")
fin = pd.read_csv("results/final_model_metrics.csv").set_index("Model")
trees = pd.read_csv("results/balancing_trees.csv")
stress = pd.read_csv("results/stress_test.csv")

f = lambda v: f"{v:.3f}"
st = [P("Risk Alert Classifier: Final Report", "title"),
      P("Predicting High-Risk customers (Risk Status = 1) from demographic, transaction and credit-behaviour data. "
        "Dataset: 4,600 customers, 17 input features, 12.1% High Risk.", "sub")]

st += [P("1. Executive summary", "h1")]
st += bullets([
    "<b>Recommended model: Tuned Logistic Regression</b> (L1 penalty, C = 0.3, class_weight = balanced, threshold 0.50). "
    "On the held-out test set (920 customers, 111 high risk) it produced <b>0 false negatives and 2 false positives</b> "
    "(recall 1.000, precision 0.982, AUC 1.000).",
    "Repeated cross-validation on the training data gives recall 0.988 +/- 0.012 and precision 0.961 for this model, the best "
    "recall among models that meet a 0.95 precision floor.",
    "Accuracy is misleading here: predicting \"Low Risk\" for everyone scores 87.9% with zero recall.",
    "Balancing techniques (under/over-sampling, SMOTE, ADASYN) did not help Logistic Regression, but raised recall for the "
    "Decision Tree (0.838 to about 0.93) and rescued the Random Forest under severe imbalance. Each trades precision for recall.",
    "The near-perfect scores indicate a very clean, probably synthetic dataset; real-world performance should be expected to be lower."])

st += [P("2. Approach", "h1")]
st += [table([
    ["Step", "What was done"],
    ["Preparation", "Dropped customer_id; converted last_transaction_date to days since last transaction; stratified 80/20 split "
                    "(High-Risk share 12.1% in train and test); one-hot encoding; standardisation; multivariate KNN imputation (k = 5) of "
                    "7 columns with 2-5% missing values, all fitted on the training set only."],
    ["Baseline", "Logistic Regression on the imbalanced data."],
    ["Imbalance", "Random under-sampling, random over-sampling, SMOTE and ADASYN applied to the training set only; compared on "
                  "Logistic Regression, Decision Tree and Random Forest, plus a severe-imbalance stress test (1.8% High Risk)."],
    ["Tree models", "Decision Tree (overfitting analysis) and Random Forest."],
    ["Tuning", "Randomized Search CV (Decision Tree, Random Forest), Grid Search CV (Random Forest and Logistic Regression); "
               "5-fold stratified CV with F2 scoring so that recall counts more than precision."],
    ["Selection", "ROC/AUC for all models, repeated 5x2 CV on the training set, rule: highest CV recall with CV precision >= 0.95."],
], [3 * cm, 14 * cm]), Spacer(1, 6)]
st += [img("01_target_and_missing", 15.5 * cm), P("Figure 1. Target distribution and missing values.", "cap")]

st += [P("3. Best classification model and justification", "h1")]
rows = [["Model (test set)", "Recall", "Precision", "F1", "AUC", "FN", "FP"]]
order = ["Logistic Regression (baseline)", "Tuned Logistic Regression (Grid)", "Tuned Random Forest (Grid)",
         "Tuned Random Forest (Randomized)", "Random Forest (default)", "Tuned Decision Tree (Randomized)", "Decision Tree (default)"]
for m in order:
    r = lb.loc[m]
    nm = f"<b>{m}</b>" if m.startswith("Tuned Logistic") else m
    rows.append([nm, f(r.Recall), f(r.Precision), f(r.F1), f"{r.AUC:.4f}", int(r["FN (Type-II)"]), int(r["FP (Type-I)"])])
st += [table(rows, [6.4 * cm, 1.9 * cm, 2.1 * cm, 1.8 * cm, 1.9 * cm, 1.4 * cm, 1.5 * cm]), Spacer(1, 6)]
rows = [["Model (5x2 repeated CV, training set)", "Recall", "Precision", "F2", "AUC"]]
for m, r in cvt.iterrows():
    nm = f"<b>{m}</b>" if m.startswith("Tuned Logistic") else m
    rows.append([nm, f"{r['CV Recall']:.3f} +/- {r['± Recall']:.3f}", f(r["CV Precision"]), f(r["CV F2"]), f"{r['CV AUC']:.4f}"])
st += [table(rows, [7.2 * cm, 3.6 * cm, 2.2 * cm, 2 * cm, 2 * cm]), Spacer(1, 6)]
st += [P("The test set contains only 111 high-risk customers, so one error changes recall by about 0.9 points; the cross-validated "
         "comparison is therefore used for the final decision. A linear model wins because the risk label is close to a weighted "
         "combination of the credit-behaviour features, so extra flexibility adds variance but no signal. It is also the most explainable "
         "option: income (negative) and late-payment days, utilisation, debt balance and missed payments (positive) are its largest drivers.")]
st += [img("10_roc_all_models", 17 * cm), P("Figure 2. ROC curves for all models (right: zoom on the top-left corner).", "cap")]

st += [P("4. Impact of imbalance-handling techniques", "h1")]
t = trees.copy()
rows = [["Base model", "Balancing", "Precision", "Recall", "F1", "FN", "FP"]]
for _, r in t.iterrows():
    rows.append([r["Base model"], r["Balancing"], f(r.Precision), f(r.Recall), f(r.F1), int(r["FN (Type-II)"]), int(r["FP (Type-I)"])])
st += [table(rows, [3.3 * cm, 3.6 * cm, 2.3 * cm, 2.2 * cm, 2 * cm, 1.8 * cm, 1.8 * cm]), Spacer(1, 4),
       P("Table: effect of balancing on tree models (test set). For Logistic Regression, recall stayed at 1.000 with 1-2 extra false "
         "positives under every technique.", "cap")]
st += [img("06b_balancing_trees", 16 * cm), P("Figure 3. Recall, F1 and AUC for the tree models under each balancing technique.", "cap")]
rows = [["Model", "Balancing", "Precision", "Recall", "F1", "FN", "FP"]]
for _, r in stress.iterrows():
    rows.append([r["Model"], r["Balancing"], f(r.Precision), f(r.Recall), f(r.F1), int(r["FN (Type-II)"]), int(r["FP (Type-I)"])])
st += [P("Stress test: training set reduced to 60 high-risk customers (1.8%).", "h2"),
       table(rows, [4 * cm, 3.4 * cm, 2.2 * cm, 2 * cm, 2 * cm, 1.7 * cm, 1.7 * cm]), Spacer(1, 6)]
st += [P("Overfitting check: the unpruned Decision Tree scores 1.000 F1 on training data but 0.869 on test data (gap 0.131); "
         "the Random Forest's gap is 0.009.")]
st += [img("08_dt_overfitting_curve", 15.5 * cm), P("Figure 4. Train vs test F1 as tree depth grows.", "cap")]

st += [P("5. Business interpretation of false positives and false negatives", "h1")]
st += [table([
    ["Error", "Meaning", "Business impact", "Final model (test)"],
    ["False negative (Type-II)", "High-risk customer predicted Low Risk",
     "Undetected default / fraud exposure: the costly error, so minimised first", f"{int(fin.iloc[0]['FN (Type-II)'])} of 111"],
    ["False positive (Type-I)", "Low-risk customer flagged High Risk",
     "Analyst review time, possible customer friction", f"{int(fin.iloc[0]['FP (Type-I)'])} of 809"],
], [3.5 * cm, 4.3 * cm, 6 * cm, 3.2 * cm]), Spacer(1, 6),
       P("Threshold analysis on out-of-fold predictions shows the default 0.50 cut-off is already F2-optimal for the class-weighted model. "
         "A lower cut-off (about 0.32) is available as a high-recall option: on the test set it caught no additional customers but doubled the false positives.")]
st += [img("13_final_confusion_matrix", 15 * cm), P("Figure 5. Final model confusion matrices on the test set.", "cap")]

st += [P("6. Conclusions and recommendations", "h1")]
st += bullets([
    "Deploy the tuned Logistic Regression (threshold 0.50) as the alerting model, with human review of each alert.",
    "Keep the lower-threshold (high-recall) option available if the business wants a bigger safety margin.",
    "When tree ensembles are used, apply class weights or under-sampling/SMOTE and report recall and precision, not accuracy.",
    "Monitor recall on confirmed outcomes and drift in late-payment days, utilisation and income; retrain regularly.",
    "Run a fairness review before real use (the Other gender group has only 43 customers) and validate on fresh, out-of-time data."])
st += [P("7. Limitations", "h1")]
st += bullets([
    "Very high scores and the absence of leakage point to a clean, probably synthetic dataset; expect lower real-world performance.",
    "Only 111 high-risk customers in the test set; differences of one or two customers are within noise.",
    "SMOTE and ADASYN results in this report come from built-in implementations of the same algorithms because imbalanced-learn was "
    "unavailable when outputs were generated; the notebook uses the library automatically when installed."])

build("docs/Final_Report.pdf", st, "Final Report - Risk Alert Classifier")
print("written docs/Final_Report.pdf")
