"""Set F, OlmoEarth v1.3 (Ai2), added on 2 Nov after the one-time test of sets A to E.

Same protocol as sets A to E, with nothing chosen anew:
  * validation: the rows, GroupKFold(5) by 5 km cell, samples and polygon-weighted scores of
    s09, with the common setting fixed in s09 (no tuning for set F);
  * test: fitted once on all training pixels (30 per polygon) and scored once on the 306
    points, with 2 000 bootstrap resamples and exact McNemar tests against sets C, D, E and
    the benchmarks, as in s12.
The test points had already been used for sets A to E, so set F's test result is reported as
a later addition. Its setting was fixed before set F existed, so the test is still its only
look at those points.

Output: prep/checks/olmoearth_cv.csv, olmoearth_test.csv, olmoearth_pairs.csv,
prep/test_predictions_F.parquet
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold

from common import FEAT, PREP, log_to
from s05_benchmarks import scores
from s09_tune import SEED, VAL_CAP, cap_rows, poly_weights, training_table, weighted_scores
from s12_test import bootstrap

log = log_to("s16_olmoearth")
CHECKS = PREP / "checks"


def rf(cfg) -> RandomForestClassifier:
    mf = cfg["max_features"]
    return RandomForestClassifier(300, max_features=float(mf) if mf != "sqrt" else mf,
                                  min_samples_leaf=int(cfg["min_samples_leaf"]), n_jobs=-1, random_state=SEED)


def main():
    meta, _ = training_table()  # the same rows as sets A to E
    cols = json.loads((FEAT / "columns.json").read_text())["F"]
    F = pd.read_parquet(FEAT / "set_F.parquet").set_index("sid")[cols]
    X = F.loc[meta.index]
    log.info("set F: %d features, training rows without values %d", len(cols), int(X.isna().all(axis=1).sum()))
    cfg = json.loads((CHECKS / "tune_best.json").read_text())["common"]

    # validation, identical design to s09
    folds = list(GroupKFold(5).split(meta, groups=meta.cell5km))
    val_rows = [cap_rows(meta.iloc[v], VAL_CAP, SEED) for _, v in folds]
    ys, ps, ws, fold_f1, tr = [], [], [], [], []
    for k, ((tr_i, _), vr) in enumerate(zip(folds, val_rows)):
        rows = cap_rows(meta.iloc[tr_i], int(cfg["cap"]), SEED + k)
        m = rf(cfg).fit(X.loc[rows], meta.label.loc[rows])
        y, p, w = meta.label.loc[vr].to_numpy(), m.predict(X.loc[vr]), poly_weights(meta.loc[vr])
        ys.append(y)
        ps.append(p)
        ws.append(w)
        fold_f1.append(weighted_scores(y, p, w)["F1_crop"])
        tr.append(weighted_scores(meta.label.loc[rows].to_numpy(), m.predict(X.loc[rows]),
                                  poly_weights(meta.loc[rows]))["F1_crop"])
    cv = dict(set="F", **weighted_scores(np.concatenate(ys), np.concatenate(ps), np.concatenate(ws)),
              F1_fold_sd=float(np.std(fold_f1)), F1_fold_min=float(np.min(fold_f1)), F1_train=float(np.mean(tr)))
    pd.DataFrame([cv]).to_csv(CHECKS / "olmoearth_cv.csv", index=False)
    log.info("validation: %s", {k: round(v, 3) if isinstance(v, float) else v for k, v in cv.items()})

    # test, identical design to s12
    rows = cap_rows(meta, int(cfg["cap"]), SEED)
    model = rf(cfg).fit(X.loc[rows], meta.label.loc[rows])
    pred = pd.read_parquet(PREP / "test_predictions.parquet").set_index("sid")
    Xt = F.loc[pred.index]
    assert not Xt.isna().all(axis=1).any(), "a test point has no OlmoEarth values"
    pred["RF F"] = model.predict(Xt)
    pred["RF F p_crop"] = model.predict_proba(Xt)[:, 1]
    pred[["test_id", "RF F", "RF F p_crop"]].reset_index().to_parquet(PREP / "test_predictions_F.parquet", index=False)

    rng = np.random.default_rng(SEED)
    res = []
    for m in ("RF F", "RF C", "RF D", "RF E"):
        for subset, sel in (("all", pred.label.notna()), ("unanimous", pred.agreement == "unanimous"),
                            ("split", pred.agreement == "split")):
            y, p = pred.label[sel].to_numpy(), pred[m][sel].to_numpy()
            res.append(dict(map=m, subset=subset, **scores(y, p), **(bootstrap(y, p, rng) if m == "RF F" else {})))
    for r, g in pred.groupby("region"):
        res.append(dict(map="RF F", subset=f"region {r}", n=len(g), called_crop=float(g["RF F"].mean()),
                        label_share=float(g.label.mean()), OA=float((g["RF F"] == g.label).mean())))
    df = pd.DataFrame(res)
    df.to_csv(CHECKS / "olmoearth_test.csv", index=False)
    log.info("\n%s", df.round(3).to_string(index=False))

    y = pred.label.to_numpy()
    pr = []
    for b in ("RF C", "RF D", "RF E", "WorldCover 2021", "DEA crop mask 2019"):
        ra, rb = pred["RF F"].to_numpy() == y, pred[b].to_numpy() == y
        a_only, b_only = int((ra & ~rb).sum()), int((~ra & rb).sum())
        pv = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
        pr.append(dict(a="RF F", b=b, a_right_b_wrong=a_only, b_right_a_wrong=b_only,
                       OA_diff=float(ra.mean() - rb.mean()), p_mcnemar=float(pv)))
    pd.DataFrame(pr).to_csv(CHECKS / "olmoearth_pairs.csv", index=False)
    log.info("\n%s", pd.DataFrame(pr).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
