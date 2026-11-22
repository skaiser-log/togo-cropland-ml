"""Step 6: error analysis on the test points (no model is refitted or changed here).

Uses prep/test_predictions.parquet from s12 and the features and benchmark values at the
test points. Main model: the set C forest (readable features) and the set D forest (the
mapped one), against WorldCover 2021 and the DEA crop mask.

1. Error overlap: per point, which of RF C, RF D, WorldCover 2021, DEA are right.
2. What WorldCover 2021 sees at each point (its own class), split by label and outcome.
3. Errors against expert votes (0-3 crop votes), region, distance to the nearest training
   polygon, and the forest's crop probability (calibration).
4. Season signals at the points: VH yearly spread, NDVI after the rains, NDVI p90.
5. Image chips (Esri World Imagery, recent, not 2019; Sentinel-2 of 2019 beside it) for a
   sample of misclassified points: prep/checks/chips/<test_id>.png

Output: prep/checks/errors_*.csv, prep/checks/chips/
"""
import io
import json
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import requests

from common import FEAT, GRID_CRS, PREP, log_to

log = log_to("s15_errors")
CHECKS = PREP / "checks"
WC_NAMES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
            60: "bare", 80: "water", 90: "wetland", 95: "mangrove", 100: "moss/lichen"}


def outcome(y, p):
    return np.select([(y == 1) & (p == 1), (y == 0) & (p == 0), (y == 0) & (p == 1)], ["TP", "TN", "FP"], "FN")


def table() -> pd.DataFrame:
    p = pd.read_parquet(PREP / "test_predictions.parquet").set_index("sid")
    b = pd.read_parquet(PREP / "benchmarks.parquet").set_index("sid").loc[p.index]
    p["wc_class"] = b.wc2021.map(WC_NAMES).fillna("none")
    cols = json.loads((FEAT / "columns.json").read_text())["C"]
    c = pd.read_parquet(FEAT / "set_C.parquet").set_index("sid").loc[p.index, cols]
    for k in ("c_vh_db_std", "b_ndvi_ond", "b_ndvi_p90", "b_ndvi_amj", "c_vh_db_p10"):
        p[k] = c[k]
    poly = gpd.read_file(PREP / "labels.gpkg", layer="train_polygons").to_crs(GRID_CRS)
    pts = gpd.GeoSeries(gpd.points_from_xy(p.lon, p.lat), crs=4326).to_crs(GRID_CRS)
    p["dist_train_km"] = pts.apply(lambda g: poly.distance(g).min()).to_numpy() / 1000
    for m in ("RF C", "RF D", "WorldCover 2021", "DEA crop mask 2019"):
        p[f"out {m}"] = outcome(p.label.to_numpy(), p[m].to_numpy())
    return p


def main():
    t = table()
    y = t.label
    right = {m: t[m] == y for m in ("RF C", "RF D", "WorldCover 2021", "DEA crop mask 2019")}

    # 1. overlap
    ov = pd.DataFrame({"models right": right["RF C"] & right["RF D"], "WC right": right["WorldCover 2021"],
                       "DEA right": right["DEA crop mask 2019"], "label": y})
    ov["pattern"] = np.select(
        [ov["models right"] & ov["WC right"] & ov["DEA right"],
         ~(right["RF C"] | right["RF D"]) & ~ov["WC right"] & ~ov["DEA right"],
         ov["models right"] & ~ov["WC right"], ~(right["RF C"] | right["RF D"]) & ov["WC right"]],
        ["all right", "all wrong", "models right, WC wrong", "models wrong, WC right"], "mixed")
    t["pattern"] = ov.pattern
    pat = pd.crosstab(ov.pattern, ov.label.map({0: "non-crop", 1: "crop"}), margins=True)
    pat.to_csv(CHECKS / "errors_overlap.csv")
    log.info("\n%s", pat)

    # 2. WorldCover class
    wc = pd.crosstab([t.label.map({0: "non-crop", 1: "crop"}), t.wc_class], t["out RF D"])
    wc.to_csv(CHECKS / "errors_wc_class.csv")
    log.info("\n%s", wc)

    # 3. votes, region, distance, calibration
    rows = []
    for m in ("RF C", "RF D", "WorldCover 2021", "DEA crop mask 2019"):
        for v, g in t.groupby("votes"):
            rows.append(dict(map=m, by="crop votes", group=str(v), n=len(g), called_crop=float(g[m].mean()),
                             accuracy=float((g[m] == g.label).mean())))
        for r, g in t.groupby("region"):
            rows.append(dict(map=m, by="region", group=r, n=len(g), called_crop=float(g[m].mean()),
                             accuracy=float((g[m] == g.label).mean())))
        for d, g in t.groupby(pd.cut(t.dist_train_km, [0, 2, 5, 10, 100])):
            rows.append(dict(map=m, by="km to training", group=str(d), n=len(g), called_crop=float(g[m].mean()),
                             accuracy=float((g[m] == g.label).mean())))
    by = pd.DataFrame(rows)
    by.to_csv(CHECKS / "errors_by_group.csv", index=False)
    log.info("\n%s", by.round(2).to_string(index=False))
    cal = t.groupby(pd.cut(t["RF D p_crop"], [0, .2, .4, .5, .6, .8, 1.0], include_lowest=True)).agg(
        n=("label", "size"), crop_share=("label", "mean"), mean_p=("RF D p_crop", "mean"))
    cal.to_csv(CHECKS / "errors_calibration_D.csv")
    log.info("\n%s", cal.round(2))

    # 4. season signals by outcome
    sig = t.groupby("out RF C")[["c_vh_db_std", "b_ndvi_ond", "b_ndvi_amj", "b_ndvi_p90", "c_vh_db_p10",
                                 "dist_train_km"]].median()
    sig["n"] = t.groupby("out RF C").size()
    sig.to_csv(CHECKS / "errors_signals_C.csv")
    log.info("\n%s", sig.round(2))

    t.reset_index().to_parquet(PREP / "test_errors.parquet", index=False)
    chips(t)


def tile_xy(lon, lat, z):
    n = 2**z
    x = (lon + 180) / 360 * n
    y = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
    return x, y


def esri_chip(lon, lat, z=17, half=1):
    from PIL import Image

    x, y = tile_xy(lon, lat, z)
    tx, ty = int(x), int(y)
    im = Image.new("RGB", (256 * (2 * half + 1),) * 2)
    for dx in range(-half, half + 1):
        for dy in range(-half, half + 1):
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{ty + dy}/{tx + dx}"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            im.paste(Image.open(io.BytesIO(r.content)).convert("RGB"), ((dx + half) * 256, (dy + half) * 256))
    px, py = (x - tx + half) * 256, (y - ty + half) * 256
    return im, px, py, 156543.03 * math.cos(math.radians(lat)) / 2**z


def chips(t: pd.DataFrame, per_group: int = 6):
    """Esri imagery for misclassified points of RF D and for points the models fix."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = CHECKS / "chips"
    out.mkdir(exist_ok=True)
    groups = {
        "D missed crop (FN)": t[t["out RF D"] == "FN"],
        "D false crop (FP)": t[t["out RF D"] == "FP"],
        "D right, WorldCover missed crop": t[(t["out RF D"] == "TP") & (t["out WorldCover 2021"] == "FN")],
    }
    rows = []
    for name, g in groups.items():
        for r in g.sample(min(per_group, len(g)), random_state=1).itertuples():
            f = out / f"{r.test_id}.png"
            if not f.exists():
                im, px, py, res = esri_chip(r.lon, r.lat)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(im)
                ax.plot(px, py, marker="s", ms=6, mfc="none", mec="yellow", mew=1.5)  # 10 m pixel at z17 is ~8 px
                ax.set_axis_off()
                fig.tight_layout(pad=0)
                fig.savefig(f, dpi=80)
                plt.close(fig)
            rows.append(dict(group=name, test_id=int(r.test_id), region=r.region, votes=int(r.votes),
                             wc_class=r.wc_class, p_crop_D=round(float(t.loc[r.Index, "RF D p_crop"]), 2),
                             file=f.name))
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)
    log.info("chips: %d", len(rows))


if __name__ == "__main__":
    main()
