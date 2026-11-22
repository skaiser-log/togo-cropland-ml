"""Smoke run of the comparison on the training data only.

Purpose: prove that every feature set loads, aligns with the labels and trains, and give
a first validation reading before the real pipeline. The 306 test points are never
loaded here; ``training_table`` refuses anything but kind == "train".

Validation design checked here:
  * GroupKFold by polygon (the plan), so pixels of one field never sit on both sides;
  * GroupKFold by 5 km cell, which also keeps neighbouring polygons together. If scores
    drop clearly from the first to the second, nearby polygons leak information and the
    final tuning should group by cell.
Pixels are capped at 30 per polygon so that a few large non-crop polygons do not dominate.

Output: prep/checks/smoke_cv.csv
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from common import FEAT, PREP

SEED = 42
CAP = 30


def training_table(set_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    smp = pd.read_parquet(PREP / "samples.parquet")
    smp = smp[smp.kind == "train"]
    assert (smp.kind == "train").all() and smp.test_id.isna().all()
    smp = smp.sample(frac=1, random_state=SEED).groupby("poly_id").head(CAP).sort_values("sid")
    cols = json.loads((FEAT / "columns.json").read_text())[set_name]
    X = pd.read_parquet(FEAT / f"set_{set_name}.parquet").set_index("sid").loc[smp.sid, cols]
    return X, smp.set_index("sid")


def scores(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    ua, pa = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(OA=float((tp + tn) / len(y)), UA_crop=float(ua), PA_crop=float(pa),
                F1_crop=float(2 * ua * pa / max(ua + pa, 1e-9)))


def main():
    rows = []
    for s in "ABCDE":
        X, meta = training_table(s)
        ok = ~X.isna().all(axis=1)
        X, meta = X[ok], meta[ok]
        y = meta.label.to_numpy()
        for grouping in ("poly_id", "cell5km"):
            rf = RandomForestClassifier(300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)
            p = cross_val_predict(rf, X, y, groups=meta[grouping], cv=GroupKFold(5))
            # per polygon weight so that every polygon counts once, as a field would
            w = 1 / meta.groupby("poly_id").poly_id.transform("size").to_numpy()
            r = dict(set=s, model="RF", grouping=grouping, n_pixels=len(y), n_features=X.shape[1],
                     nan_share=float(X.isna().mean().mean()), **scores(y, p),
                     OA_poly_weighted=float(np.sum(w * (p == y)) / w.sum()))
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    best = max((r for r in rows if r["grouping"] == "poly_id"), key=lambda r: r["F1_crop"])["set"]
    X, meta = training_table(best)
    X = X[~X.isna().all(axis=1)]
    meta = meta.loc[X.index]
    for depth in (3, 6, None):
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=20, random_state=SEED)
        p = cross_val_predict(dt, X, meta.label, groups=meta.poly_id, cv=GroupKFold(5))
        r = dict(set=best, model=f"tree depth {depth}", grouping="poly_id", n_pixels=len(X),
                 n_features=X.shape[1], **scores(meta.label.to_numpy(), p))
        rows.append(r)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(PREP / "checks/smoke_cv.csv", index=False)


if __name__ == "__main__":
    main()
