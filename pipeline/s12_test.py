"""Step 4: the one-time test on the 306 random test points.

Every model is fitted once on all training pixels (at most 30 per polygon, the common
setting from s09) and predicts the test points once. Nothing here feeds back into a
setting: the choices were fixed on the validation folds before this script was run.

Models: Random Forest on sets A to E; the depth-6 decision tree on set C (the best tree
depth in s10). Benchmarks: WorldCover 2020 and 2021, DEA crop mask 2019, on exactly the
same points.

Scores for crop (OA, UA, PA, F1) on all, unanimous and split points and per zone, with
95 % bootstrap intervals (2 000 resamples of the test points). Paired comparisons use the
exact McNemar test on the points where two maps disagree.

Output: prep/checks/test_scores.csv, test_scores_zone.csv, test_pairs.csv,
prep/test_predictions.parquet (one row per test point, every model and benchmark)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from common import FEAT, PREP, log_to
from s05_benchmarks import scores
from s09_tune import SEED, SETS, cap_rows, training_table

log = log_to("s12_test")
CHECKS = PREP / "checks"
B = 2000


def test_table() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    smp = pd.read_parquet(PREP / "samples.parquet")
    t = smp[smp.kind == "test"].set_index("sid")
    cols = json.loads((FEAT / "columns.json").read_text())
    X = {s: pd.read_parquet(FEAT / f"set_{s}.parquet").set_index("sid").loc[t.index, cols[s]] for s in SETS}
    for s in SETS:
        assert not X[s].isna().all(axis=1).any(), f"test point without values in set {s}"
    return t, X


def bootstrap(y, p, rng) -> dict:
    n = len(y)
    idx = rng.integers(0, n, size=(B, n))
    out = {k: [] for k in ("OA", "UA_crop", "PA_crop", "F1_crop")}
    for i in idx:
        sc = scores(y[i], p[i])
        for k in out:
            out[k].append(sc[k])
    return {f"{k}_lo": float(np.percentile(v, 2.5)) for k, v in out.items()} | \
           {f"{k}_hi": float(np.percentile(v, 97.5)) for k, v in out.items()}


def main():
    meta, Xtr = training_table()
    cfg = json.loads((CHECKS / "tune_best.json").read_text())["common"]
    rows = cap_rows(meta, int(cfg["cap"]), SEED)
    ytr = meta.label.loc[rows]
    log.info("training on %d pixels from %d polygons, crop share %.2f", len(rows),
             meta.loc[rows].poly_id.nunique(), ytr.mean())
    t, Xte = test_table()

    pred = pd.DataFrame(index=t.index)
    for s in SETS:
        rf = RandomForestClassifier(300, max_features=float(cfg["max_features"]) if cfg["max_features"] != "sqrt"
                                    else "sqrt", min_samples_leaf=int(cfg["min_samples_leaf"]),
                                    n_jobs=-1, random_state=SEED).fit(Xtr[s].loc[rows], ytr)
        pred[f"RF {s}"] = rf.predict(Xte[s])
        pred[f"RF {s} p_crop"] = rf.predict_proba(Xte[s])[:, 1]
    tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=SEED).fit(Xtr["C"].loc[rows], ytr)
    pred["Tree C depth 6"] = tree.predict(Xte["C"])

    bench = pd.read_parquet(PREP / "benchmarks.parquet").set_index("sid").loc[t.index]
    pred["WorldCover 2020"] = (bench.wc2020 == 40).astype(int)
    pred["WorldCover 2021"] = (bench.wc2021 == 40).astype(int)
    pred["DEA crop mask 2019"] = (bench.dea_mask.fillna(0) == 1).astype(int)  # 255 = masked, non-crop

    maps = [c for c in pred.columns if not c.endswith("p_crop")]
    out = t[["test_id", "label", "votes", "agreement", "zone", "region", "lon", "lat"]].join(pred)
    out.reset_index().to_parquet(PREP / "test_predictions.parquet", index=False)

    rng = np.random.default_rng(SEED)
    res, zres = [], []
    for m in maps:
        for subset, sel in (("all", out.label.notna()), ("unanimous", out.agreement == "unanimous"),
                            ("split", out.agreement == "split")):
            y, p = out.label[sel].to_numpy(), out[m][sel].to_numpy()
            res.append(dict(map=m, subset=subset, **scores(y, p), **bootstrap(y, p, rng)))
        for z, g in out.groupby("zone"):
            zres.append(dict(map=m, zone=z, **scores(g.label.to_numpy(), g[m].to_numpy())))
    df = pd.DataFrame(res)
    df.to_csv(CHECKS / "test_scores.csv", index=False)
    pd.DataFrame(zres).to_csv(CHECKS / "test_scores_zone.csv", index=False)
    log.info("\n%s", df[df.subset == "all"][["map", "OA", "OA_lo", "OA_hi", "UA_crop", "PA_crop", "F1_crop",
                                                 "F1_crop_lo", "F1_crop_hi"]].round(3).to_string(index=False))

    pairs = [("RF B", "RF A"), ("RF C", "RF B"), ("RF D", "RF C"), ("RF E", "RF C"), ("RF D", "RF E"),
             ("RF C", "Tree C depth 6"), ("RF C", "WorldCover 2021"), ("RF D", "WorldCover 2021"),
             ("RF E", "WorldCover 2021"), ("RF D", "DEA crop mask 2019"), ("RF E", "DEA crop mask 2019")]
    pr = []
    y = out.label.to_numpy()
    for a, b in pairs:
        ra, rb = out[a].to_numpy() == y, out[b].to_numpy() == y
        only_a, only_b = int((ra & ~rb).sum()), int((~ra & rb).sum())
        pv = binomtest(only_a, only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
        pr.append(dict(a=a, b=b, a_right_b_wrong=only_a, b_right_a_wrong=only_b,
                       OA_diff=float(ra.mean() - rb.mean()), p_mcnemar=float(pv)))
    pd.DataFrame(pr).to_csv(CHECKS / "test_pairs.csv", index=False)
    log.info("\n%s", pd.DataFrame(pr).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
