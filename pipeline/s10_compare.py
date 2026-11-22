"""Step 2 of the modelling: decision tree against Random Forest, and what set C relies on.

Uses the folds, samples and scores of ``s09_tune`` (training data only) and the common
Random Forest setting in ``prep/checks/tune_best.json``.

1. Decision tree against forest on set C (hand-built, readable) and on the best embedding.
   Depths 3, 4, 6, 8 and unlimited, min_samples_leaf 20, same cap as the forest.
2. Grouped permutation importance of the set C forest on the validation folds. Columns
   are permuted together per group (radar by season, optical by kind), because single
   columns are strongly correlated and would each look unimportant. Importance is the
   drop in polygon-weighted F1 crop, averaged over folds and three repeats.
3. The depth-3 tree on set C, fitted on all training pixels, as text rules for the poster.

Output: prep/checks/compare_tree_rf.csv, perm_importance_C.csv, tree_rules_C.txt
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from common import PREP, log_to
from s09_tune import SEED, VAL_CAP, cap_rows, poly_weights, training_table, weighted_scores

log = log_to("s10_compare")
CHECKS = PREP / "checks"
SEASONS = {"DJF": ("12", "01", "02"), "MAM": ("03", "04", "05"),
           "JJA": ("06", "07", "08"), "SON": ("09", "10", "11")}


def groups_C(cols: list[str]) -> dict[str, list[str]]:
    g = {
        "S2 band percentiles": [c for c in cols if c.startswith("b_B")],
        "S2 index percentiles": [c for c in cols if c.startswith("b_n") and c.rsplit("_", 1)[1][0] == "p"
                                 or c.endswith("_std") and c.startswith("b_")],
        "S2 seasonal NDVI/NDMI": [c for c in cols if c.rsplit("_", 1)[1] in ("djf", "amj", "jas", "ond")],
        "S2 NDVI harmonics + range": [c for c in cols if "_h_" in c or c.endswith("_range")],
        "S1 annual stats": [c for c in cols if c.startswith("c_") and "_m" not in c[-4:]],
    }
    for name, months in SEASONS.items():
        g[f"S1 monthly {name}"] = [c for c in cols if c.startswith("c_") and c[-3:] in {f"m{m}" for m in months}]
    used = sum(g.values(), [])
    assert sorted(used) == sorted(cols) and len(used) == len(set(used)), set(cols) ^ set(used)
    return g


def rf_for(best: dict) -> RandomForestClassifier:
    mf = best["max_features"]
    mf = float(mf) if mf not in ("sqrt", "log2") else mf
    return RandomForestClassifier(300, max_features=mf, min_samples_leaf=int(best["min_samples_leaf"]),
                                  n_jobs=-1, random_state=SEED)


def main():
    meta, X = training_table()
    best = json.loads((CHECKS / "tune_best.json").read_text())
    folds = list(GroupKFold(5).split(meta, groups=meta.cell5km))
    val_rows = [cap_rows(meta.iloc[v], VAL_CAP, SEED) for _, v in folds]
    emb = max("DE", key=lambda s: best[s]["F1_crop"])
    cfg = best["common"]  # one setting for all sets, chosen in s09

    def cv(make, s, cap):
        ys, ps, ws, tr = [], [], [], []
        for k, ((tr_i, _), vr) in enumerate(zip(folds, val_rows)):
            rows = cap_rows(meta.iloc[tr_i], cap, SEED + k)
            m = make().fit(X[s].loc[rows], meta.label.loc[rows])
            ys.append(meta.label.loc[vr].to_numpy())
            ps.append(m.predict(X[s].loc[vr]))
            ws.append(poly_weights(meta.loc[vr]))
            tr.append(weighted_scores(meta.label.loc[rows].to_numpy(), m.predict(X[s].loc[rows]),
                                      poly_weights(meta.loc[rows]))["F1_crop"])
        return dict(**weighted_scores(np.concatenate(ys), np.concatenate(ps), np.concatenate(ws)),
                    F1_train=float(np.mean(tr)))

    rows = []
    for s in ("C", emb):
        cap = int(cfg["cap"])
        r = dict(set=s, model="Random Forest", depth="", **cv(lambda: rf_for(cfg), s, cap))
        rows.append(r)
        log.info("%s", r)
        for d in (3, 4, 6, 8, None):
            r = dict(set=s, model="Decision tree", depth=str(d), **cv(
                lambda: DecisionTreeClassifier(max_depth=d, min_samples_leaf=20, random_state=SEED), s, cap))
            rows.append(r)
            log.info("%s", r)
    pd.DataFrame(rows).to_csv(CHECKS / "compare_tree_rf.csv", index=False)

    # grouped permutation importance, set C
    groups = groups_C(list(X["C"].columns))
    rng = np.random.default_rng(SEED)
    imp = {g: [] for g in groups}
    cap = int(cfg["cap"])
    for k, ((tr_i, _), vr) in enumerate(zip(folds, val_rows)):
        rows_k = cap_rows(meta.iloc[tr_i], cap, SEED + k)
        m = rf_for(cfg).fit(X["C"].loc[rows_k], meta.label.loc[rows_k])
        Xv, y, w = X["C"].loc[vr], meta.label.loc[vr].to_numpy(), poly_weights(meta.loc[vr])
        base = weighted_scores(y, m.predict(Xv), w)["F1_crop"]
        for g, cols in groups.items():
            for _ in range(3):
                Xp = Xv.copy()
                Xp[cols] = Xv[cols].to_numpy()[rng.permutation(len(Xv))]
                imp[g].append(base - weighted_scores(y, m.predict(Xp), w)["F1_crop"])
        log.info("importance fold %d done", k)
    pi = pd.DataFrame([dict(group=g, n_columns=len(groups[g]), F1_drop=float(np.mean(v)),
                            F1_drop_sd=float(np.std(v))) for g, v in imp.items()])
    pi.sort_values("F1_drop", ascending=False).to_csv(CHECKS / "perm_importance_C.csv", index=False)
    log.info("\n%s", pi.sort_values("F1_drop", ascending=False).to_string())

    # readable rules
    rows_all = cap_rows(meta, cap, SEED)
    t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=SEED)
    t.fit(X["C"].loc[rows_all], meta.label.loc[rows_all])
    (CHECKS / "tree_rules_C.txt").write_text(
        export_text(t, feature_names=list(X["C"].columns), class_names=["non-crop", "crop"],
                    decimals=2, show_weights=True))
    log.info("rules written")


if __name__ == "__main__":
    main()
