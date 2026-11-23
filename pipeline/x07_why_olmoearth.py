"""Exploratory: why does the OlmoEarth forest (set F) beat the season and the other embeddings?

Each hypothesis is an intervention on the inputs, with the common Random Forest setting of s09:
  dimensions     only the first 64 OlmoEarth values (its Matryoshka 64-d embedding)
  classifier     logistic regression and cosine kNN on the same embeddings
  spatial noise  every pixel replaced by the mean embedding of its polygon
  map knowledge  the WorldCover 2021 class added to each input

Scored on training polygons with GroupKFold(5) by 5 km cell and leave-one-region-out. The
map-knowledge variants are also fitted once on all training pixels and scored on the 306 test
points; these are diagnostics of a finished comparison and select nothing.

Output: prep/checks/why_cv.csv, why_test.csv, why_probe.csv
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

from common import FEAT, PREP, log_to
from s05_benchmarks import scores
from s09_tune import SEED, VAL_CAP, cap_rows, poly_weights, training_table, weighted_scores

log = log_to("x07_why_olmoearth")
CHECKS = PREP / "checks"
TREES = 150  # diagnostics; the reported models keep 300
WC_CLASSES = (10, 20, 30, 40, 50, 60, 80, 90)
warnings.filterwarnings("ignore", category=UserWarning)


def rf(cfg):
    return RandomForestClassifier(TREES, max_features=float(cfg["max_features"]),
                                  min_samples_leaf=int(cfg["min_samples_leaf"]), n_jobs=8, random_state=SEED)


def logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))


def knn():
    return make_pipeline(Normalizer(), KNeighborsClassifier(15, metric="cosine", n_jobs=8))


def wc_onehot(sids) -> pd.DataFrame:
    w = pd.read_parquet(PREP / "benchmarks.parquet").set_index("sid").loc[sids, "wc2021"]
    return pd.DataFrame({f"wc_{c}": (w == c).astype("int8").to_numpy() for c in WC_CLASSES}, index=sids)


def splits(meta):
    yield from (("cv5km", f"fold{k}", tr, va) for k, (tr, va) in
                enumerate(GroupKFold(5).split(meta, groups=meta.cell5km)))
    for r in sorted(meta.region.unique()):
        m = (meta.region == r).to_numpy()
        yield "loro", r, np.flatnonzero(~m), np.flatnonzero(m)


def evaluate(name, X, make, meta, cfg, done, out):
    rows = []
    for scheme, part, tr, va in splits(meta):
        if (name, scheme, part) in done:
            continue
        fit = cap_rows(meta.iloc[tr], int(cfg["cap"]), SEED)
        vr = cap_rows(meta.iloc[va], VAL_CAP, SEED)
        m = make().fit(X.loc[fit], meta.label.loc[fit])
        y, p, w = meta.label.loc[vr].to_numpy(), m.predict(X.loc[vr]), poly_weights(meta.loc[vr])
        rows.append(dict(variant=name, scheme=scheme, part=part, **weighted_scores(y, p, w)))
    if rows:
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)
    log.info("%s done", name)


def main():
    meta, X0 = training_table()
    cols = json.loads((FEAT / "columns.json").read_text())
    F = pd.read_parquet(FEAT / "set_F.parquet").set_index("sid")[cols["F"]]
    allx = {"B": pd.read_parquet(FEAT / "set_B.parquet").set_index("sid")[cols["B"]],
            "D": pd.read_parquet(FEAT / "set_D.parquet").set_index("sid")[cols["D"]],
            "E": pd.read_parquet(FEAT / "set_E.parquet").set_index("sid")[cols["E"]], "F": F}
    X = {"B": X0["B"], "D": X0["D"], "E": X0["E"], "F": F.loc[meta.index]}
    cfg = json.loads((CHECKS / "tune_best.json").read_text())["common"]
    wc = wc_onehot(meta.index)
    out = CHECKS / "why_cv.csv"
    done = set()
    if out.exists():
        d = pd.read_csv(out)
        done = set(zip(d.variant, d.scheme, d.part))

    for s in "BDEF":
        evaluate(f"{s} forest", X[s], lambda: rf(cfg), meta, cfg, done, out)
    evaluate("F first 64 values", X["F"].iloc[:, :64], lambda: rf(cfg), meta, cfg, done, out)
    for s in "DEF":
        evaluate(f"{s} logistic", X[s], logreg, meta, cfg, done, out)
        evaluate(f"{s} kNN", X[s], knn, meta, cfg, done, out)
        pm = X[s].groupby(meta.poly_id).transform("mean")
        evaluate(f"{s} polygon mean", pm, lambda: rf(cfg), meta, cfg, done, out)
    for s in "BDEF":
        evaluate(f"{s} + WorldCover", X[s].join(wc), lambda: rf(cfg), meta, cfg, done, out)
    evaluate("WorldCover only", wc, lambda: rf(cfg), meta, cfg, done, out)

    # diagnostics on the test points
    pred = pd.read_parquet(PREP / "test_predictions.parquet").set_index("sid")
    y = pred.label.astype(int).to_numpy()
    fit = cap_rows(meta, int(cfg["cap"]), SEED)
    wct = wc_onehot(pred.index)
    res, calls = [], pd.DataFrame(index=pred.index)
    for s in "BDEF":
        for suffix, tr, te in ((" alone", X[s], allx[s].loc[pred.index]),
                               (" + WorldCover", X[s].join(wc), allx[s].loc[pred.index].join(wct))):
            p = rf(cfg).fit(tr.loc[fit], meta.label.loc[fit]).predict(te)
            calls[s + suffix] = p
            res.append(dict(variant=s + suffix, **scores(y, p)))
    pd.DataFrame(res).to_csv(CHECKS / "why_test.csv", index=False)
    calls.reset_index().to_parquet(PREP / "why_test_calls.parquet", index=False)

    # how much of WorldCover's cropland class can each embedding read out (linear probe)
    probe = []
    wc_crop = wc["wc_40"]
    for s in "BDEF":
        ps = np.full(len(meta), np.nan)
        xs = X[s].fillna(X[s].median())
        for tr, va in GroupKFold(5).split(meta, groups=meta.cell5km):
            fit_i = meta.iloc[tr].sample(min(15000, len(tr)), random_state=SEED).index
            m = logreg().fit(xs.loc[fit_i], wc_crop.loc[fit_i])
            ps[va] = m.predict_proba(xs.iloc[va])[:, 1]
        probe.append(dict(set=s, r2_wc_crop=r2_score(wc_crop, ps), agree=float(((ps > .5) == wc_crop).mean())))
    pd.DataFrame(probe).to_csv(CHECKS / "why_probe.csv", index=False)
    log.info("finished")


if __name__ == "__main__":
    main()
