"""Tune the Random Forest per feature set on spatially grouped validation folds.

Design (training data only; ``training_table`` refuses the test points):
  * Folds: GroupKFold(5) by ``cell5km``. The smoke run showed that folds by polygon let
    neighbouring fields leak information, so whole 5 km cells stay on one side.
  * The same rows and the same folds for every feature set: pixels that are empty in any
    set (no clear Oct-Dec view for set A, or never clear) are dropped everywhere, so the
    sets are compared on identical data.
  * Training sample: at most ``cap`` pixels per polygon, drawn at random. The cap is tuned,
    because it also sets the class mix (large non-crop polygons dominate without it).
  * Validation sample: always at most 30 pixels per polygon, and every polygon weighs 1 in
    the scores, as one field counts once. All settings are scored on the same pixels.
  * Grid: cap {30, 100} x max_features {sqrt, 0.3} x min_samples_leaf {1, 5, 20},
    300 trees. Scores on the training sample are kept too, to show over- or underfitting.

Output: prep/checks/tune_cv.csv (one row per set and setting, written as it goes, so an
interrupted run resumes) and prep/checks/tune_best.json (best setting per set by F1 crop).
"""
import itertools
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from common import FEAT, PREP, log_to

SEED = 42
VAL_CAP = 30
SETS = "ABCDE"
GRID = [dict(cap=c, max_features=m, min_samples_leaf=leaf)
        for c, m, leaf in itertools.product((30, 100), ("sqrt", 0.3), (1, 5, 20))]
OUT = PREP / "checks/tune_cv.csv"
log = log_to("s09_tune")


def training_table() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Training pixels with values in every set, and the feature table per set."""
    smp = pd.read_parquet(PREP / "samples.parquet")
    smp = smp[smp.kind == "train"]
    assert (smp.kind == "train").all() and smp.test_id.isna().all()
    cols = json.loads((FEAT / "columns.json").read_text())
    X = {s: pd.read_parquet(FEAT / f"set_{s}.parquet").set_index("sid").loc[smp.sid, cols[s]]
         for s in SETS}
    ok = np.logical_and.reduce([~X[s].isna().all(axis=1).to_numpy() for s in SETS])
    smp = smp[ok].set_index("sid")
    log.info("training pixels %d of %d kept (values in every set), %d polygons",
             ok.sum(), len(ok), smp.poly_id.nunique())
    return smp, {s: X[s].loc[smp.index] for s in SETS}


def cap_rows(meta: pd.DataFrame, cap: int, seed: int) -> pd.Index:
    return meta.sample(frac=1, random_state=seed).groupby("poly_id").head(cap).index


def weighted_scores(y, p, w) -> dict:
    tp = w[(y == 1) & (p == 1)].sum()
    fp = w[(y == 0) & (p == 1)].sum()
    fn = w[(y == 1) & (p == 0)].sum()
    tn = w[(y == 0) & (p == 0)].sum()
    ua, pa = tp / max(tp + fp, 1e-9), tp / max(tp + fn, 1e-9)
    return dict(OA=(tp + tn) / w.sum(), UA_crop=ua, PA_crop=pa,
                F1_crop=2 * ua * pa / max(ua + pa, 1e-9))


def poly_weights(meta: pd.DataFrame) -> np.ndarray:
    return 1 / meta.groupby("poly_id").poly_id.transform("size").to_numpy()


def main():
    meta, X = training_table()
    folds = list(GroupKFold(5).split(meta, groups=meta.cell5km))
    val_rows = [cap_rows(meta.iloc[v], VAL_CAP, SEED) for _, v in folds]
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=["set", "setting"])
    rows = done.to_dict("records")
    for s in SETS:
        for g in GRID:
            key = json.dumps(g)
            if ((done.set == s) & (done.setting == key)).any():
                continue
            t0 = time.time()
            ys, ps, ws, fold_f1, tr = [], [], [], [], []
            for k, ((tr_i, _), vr) in enumerate(zip(folds, val_rows)):
                tr_rows = cap_rows(meta.iloc[tr_i], g["cap"], SEED + k)
                rf = RandomForestClassifier(300, max_features=g["max_features"],
                                            min_samples_leaf=g["min_samples_leaf"],
                                            n_jobs=-1, random_state=SEED)
                rf.fit(X[s].loc[tr_rows], meta.label.loc[tr_rows])
                y, p = meta.label.loc[vr].to_numpy(), rf.predict(X[s].loc[vr])
                w = poly_weights(meta.loc[vr])
                ys.append(y)
                ps.append(p)
                ws.append(w)
                fold_f1.append(weighted_scores(y, p, w)["F1_crop"])
                yt = meta.label.loc[tr_rows].to_numpy()
                tr.append(weighted_scores(yt, rf.predict(X[s].loc[tr_rows]),
                                          poly_weights(meta.loc[tr_rows]))["F1_crop"])
            sc = weighted_scores(np.concatenate(ys), np.concatenate(ps), np.concatenate(ws))
            r = dict(set=s, setting=key, **g, n_features=X[s].shape[1], **sc,
                     F1_fold_sd=float(np.std(fold_f1)), F1_fold_min=float(np.min(fold_f1)),
                     F1_train=float(np.mean(tr)), seconds=round(time.time() - t0, 1))
            rows.append(r)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            log.info("%s %s F1 %.3f (sd %.3f) train %.3f, %.0f s", s, key, sc["F1_crop"],
                     r["F1_fold_sd"], r["F1_train"], r["seconds"])
    df = pd.DataFrame(rows)
    best = {s: df[df.set == s].sort_values("F1_crop", ascending=False).iloc[0][
        ["cap", "max_features", "min_samples_leaf", "F1_crop", "F1_fold_sd", "F1_train"]].to_dict()
        for s in SETS}
    # One setting for every set keeps the comparison about the features alone. Take the
    # setting with the highest mean F1 over the sets; record how far it lies from each best.
    df["gap"] = df.groupby("set").F1_crop.transform("max") - df.F1_crop
    agg = df.groupby("setting").agg(mean_F1=("F1_crop", "mean"), worst_gap=("gap", "max"))
    key = agg.mean_F1.idxmax()
    best["common"] = dict(**json.loads(key), mean_F1=float(agg.mean_F1[key]),
                          worst_gap=float(agg.worst_gap[key]))
    (PREP / "checks/tune_best.json").write_text(json.dumps(best, indent=2, default=str))
    log.info("best per set: %s", best)


if __name__ == "__main__":
    main()
