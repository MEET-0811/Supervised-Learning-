"""Score raw customer rows with the saved final model.

Usage:
    python src/score_new_customers.py data/risk_alert_classifier_dataset.csv  [output.csv]

Input must have the same columns as the original CSV (risk_status is optional).
Output columns: customer_id, risk_probability, high_risk_alert (1 = flag for review).
"""
import sys
import joblib
import numpy as np
import pandas as pd

CAT_COLS = ["gender", "region", "employment_type"]
NUM_COLS = ["age", "annual_income_inr", "credit_score", "credit_utilization_ratio", "missed_payments_12m",
            "avg_late_payment_days", "monthly_transaction_count", "monthly_spend_inr", "cash_advance_count_6m",
            "complaints_last_6m", "failed_login_attempts_3m", "account_tenure_months", "debt_balance_inr",
            "days_since_last_txn"]


def load_bundle(path="models/risk_alert_model.joblib"):
    return joblib.load(path)


def score_customers(raw: pd.DataFrame, bundle=None) -> pd.DataFrame:
    bundle = bundle or load_bundle()
    raw = raw.copy()
    raw["days_since_last_txn"] = (bundle["ref_date"] - pd.to_datetime(raw["last_transaction_date"])).dt.days

    parts = [raw[NUM_COLS].copy()]
    for c in CAT_COLS:
        d = pd.get_dummies(raw[c], prefix=c, dtype=float)
        d.loc[raw[c].isna(), :] = np.nan
        parts.append(d)
    enc = pd.concat(parts, axis=1).reindex(columns=bundle["feature_columns"], fill_value=0.0)

    arr = bundle["knn_imputer"].transform(bundle["scaler_imputer"].transform(enc))
    arr = bundle["scaler_imputer"].inverse_transform(arr)
    feats = pd.DataFrame(arr, columns=enc.columns, index=enc.index)
    for c in CAT_COLS:
        grp = [k for k in feats.columns if k.startswith(c + "_")]
        feats[grp] = np.eye(len(grp))[feats[grp].values.argmax(axis=1)]
    feats[["age", "credit_score"]] = feats[["age", "credit_score"]].round()
    feats["credit_utilization_ratio"] = feats["credit_utilization_ratio"].clip(0, 1)

    p = bundle["model"].predict_proba(feats)[:, 1]
    return pd.DataFrame({"customer_id": raw["customer_id"].values, "risk_probability": p.round(4),
                         "high_risk_alert": (p >= bundle["threshold"]).astype(int)})


if __name__ == "__main__":
    out = score_customers(pd.read_csv(sys.argv[1]))
    if len(sys.argv) > 2:
        out.to_csv(sys.argv[2], index=False)
        print(f"wrote {len(out)} rows to {sys.argv[2]}; {int(out.high_risk_alert.sum())} flagged")
    else:
        print(out.head(10).to_string(index=False))
