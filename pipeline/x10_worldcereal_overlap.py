"""Side check: does OlmoEarth's lead depend on training polygons that WorldCereal holds?

1 274 of the 1 276 training polygons are in WorldCereal's reference repository (collection
2019_af_nhicropharvest_poly_100); 311 of them carry extract > 0, meaning they were selected for
model-input extraction. Leave-one-region-out forests are scored separately on selected and
unselected polygons. A lead caused by leakage should shrink on the unselected ones.

Input: prep/checks/worldcereal_nhi_togo.gpkg  Output: prep/checks/worldcereal_overlap.csv
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd

from common import FEAT, PREP
from s09_tune import SEED, VAL_CAP, cap_rows, poly_weights, training_table, weighted_scores
from x07_why_olmoearth import CHECKS, rf


def main():
    meta, X0 = training_table()
    cols = json.loads((FEAT / "columns.json").read_text())
    X = {"B": X0["B"], "D": X0["D"], "E": X0["E"],
         "F": pd.read_parquet(FEAT / "set_F.parquet").set_index("sid")[cols["F"]].loc[meta.index]}
    cfg = json.loads((CHECKS / "tune_best.json").read_text())["common"]
    wc = gpd.read_file(CHECKS / "worldcereal_nhi_togo.gpkg")
    wc = wc[wc.extract > 0][["geometry"]].drop_duplicates("geometry").to_crs(4326)
    s = pd.read_parquet(PREP / "samples.parquet").set_index("sid").loc[meta.index, ["lon", "lat"]]
    pts = gpd.GeoDataFrame(index=s.index, geometry=gpd.points_from_xy(s.lon, s.lat), crs=4326)
    hit = gpd.sjoin(pts, wc, predicate="within").index
    chosen = set(meta.loc[meta.index.isin(hit), "poly_id"])
    group = meta.poly_id.isin(chosen).map({True: "selected", False: "not selected"})
    print("polygons", meta.groupby(group).poly_id.nunique().to_dict(), flush=True)
    rows = []
    for k in "BDEF":
        parts = []
        for r in sorted(meta.region.unique()):
            m = (meta.region == r).to_numpy()
            fit = cap_rows(meta.iloc[np.flatnonzero(~m)], int(cfg["cap"]), SEED)
            vr = cap_rows(meta.iloc[np.flatnonzero(m)], VAL_CAP, SEED)
            p = rf(cfg).fit(X[k].loc[fit], meta.label.loc[fit]).predict(X[k].loc[vr])
            parts.append(pd.DataFrame(dict(y=meta.label.loc[vr].to_numpy(), p=p, w=poly_weights(meta.loc[vr]),
                                           g=group.loc[vr].to_numpy(), region=r)))
        d = pd.concat(parts)
        for g, x in d.groupby("g"):
            rows.append(dict(model=k, group=g, polygons=meta.loc[group == g, "poly_id"].nunique(),
                             **weighted_scores(x.y.to_numpy(), x.p.to_numpy(), x.w.to_numpy())))
        print(k, "done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(CHECKS / "worldcereal_overlap.csv", index=False)
    print(out.pivot(index="model", columns="group", values="F1_crop").round(3))


if __name__ == "__main__":
    main()
