"""Train one model per disease and save it to models/.
Run:  python train.py
"""
import json
import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import DISEASES


def load_data(name, cfg):
    if cfg["file"] is None:
        bc = load_breast_cancer(as_frame=True)
        return bc.frame
    return pd.read_csv(cfg["file"])


def main():
    all_metrics = {}
    for name, cfg in DISEASES.items():
        try:
            df = load_data(name, cfg)
        except FileNotFoundError:
            print(f"[SKIP] {name}: {cfg['file']} not found (put the CSV in the data/ folder)")
            continue

        feats = list(cfg["features"].keys())
        X, y = df[feats], df[cfg["target"]]
        # make "1" always mean "disease present"
        y = (y == cfg["positive_label"]).astype(int)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        candidates = {
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        }
        best = None
        results = {}
        for mname, model in candidates.items():
            model.fit(X_tr, y_tr)
            prob = model.predict_proba(X_te)[:, 1]
            pred = model.predict(X_te)
            m = {
                "accuracy": round(accuracy_score(y_te, pred), 4),
                "f1": round(f1_score(y_te, pred), 4),
                "roc_auc": round(roc_auc_score(y_te, prob), 4),
            }
            results[mname] = m
            if best is None or m["roc_auc"] > best[2]["roc_auc"]:
                best = (mname, model, m)

        mname, model, m = best
        # model-agnostic feature importance
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        else:
            imp = abs(model[-1].coef_[0])
            imp = imp / imp.sum()
        importance = dict(zip(feats, [float(i) for i in imp]))

        joblib.dump(
            {"model": model, "features": feats, "importance": importance, "best_model": mname},
            f"models/{name.replace(' ', '_').lower()}.joblib",
        )
        all_metrics[name] = {"best_model": mname, "results": results, "rows": int(len(df))}
        print(f"[OK] {name}: best = {mname}  {m}")

    with open("models/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)


if __name__ == "__main__":
    main()
