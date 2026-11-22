"""Step 5, part 2: cropland areas of Savanes and Plateaux, mapped and error-adjusted.

1. Mapped area: pixel count of each map inside the region (set D map from s13, set F map
   from s17, WorldCover 2020 and 2021, DEA crop mask 2019, all on the same 10 m grid).
2. Error-adjusted area (Olofsson et al. 2014, stratified estimator). The test points are a
   simple random sample of Togo, so the points inside a region are a random sample of it.
   With the map classes as strata h (weight W_h = mapped share of class h), the crop share
   is p = sum_h W_h * n_h,crop / n_h, with variance sum_h W_h^2 * p_h (1 - p_h) / (n_h - 1).
   Every map gives an unbiased estimate; a better map gives a narrower interval. The plain
   share of crop points is reported next to it.
3. Agreement with WorldCover 2021 per pixel, for each forest map m (d, f): m_vs_wc_<region>.tif
   with 0 both non-crop, 1 crop only in the forest map ("fields WorldCover misses"), 2 crop
   only in WorldCover, 3 both crop, 255 outside.
4. Consistency: each forest map read at the test points must reproduce its test predictions
   (s12 for set D, s16 for set F).

Output: prep/checks/areas.csv, prep/checks/d_vs_wc.csv (both forest maps, column model),
prep/maps/{d,f}_vs_wc_<region>.tif, prep/checks/map_<region>.png (set D) and
map_<region>_f.png (set F) quick-look figures
"""
import numpy as np
import pandas as pd
import rasterio

from common import GRID_CRS, PREP, log_to, to_crs

log = log_to("s14_areas")
MAPS = PREP / "maps"
CHECKS = PREP / "checks"
REGIONS = ("Savanes", "Plateaux")
FOREST = {"d_crop": ("RF D", "set D", "d"), "f_crop": ("RF F", "set F", "f")}


def load(name: str, region: str) -> tuple[np.ndarray, rasterio.Affine]:
    with rasterio.open(MAPS / f"{name}_{region}.tif") as s:
        return s.read(1), s.transform


def crop_mask(name: str, a: np.ndarray) -> np.ndarray:
    """1 crop, 0 non-crop, 255 outside or no data, for every map's own coding."""
    if name.startswith("wc"):
        return np.where(a == 255, 255, (a == 40).astype("uint8"))
    if name.startswith("dea"):
        return np.where(a == 255, 0, (a == 1).astype("uint8"))  # 255 inside DEA tiles = masked = non-crop
    return a


def at_points(arr, tr, lon, lat) -> np.ndarray:
    x, y = to_crs(lon, lat, "EPSG:4326", GRID_CRS)
    inv = ~tr
    c, r = inv @ (np.asarray(x), np.asarray(y))
    r, c = np.floor(r).astype(int), np.floor(c).astype(int)
    ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    v = np.full(len(r), 255, "uint8")
    v[ok] = arr[r[ok], c[ok]]
    return v


def stratified(W: dict, y: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    p, var = 0.0, 0.0
    for h, w in W.items():
        yh = y[m == h]
        if len(yh) == 0:
            return np.nan, np.nan  # a stratum without points: estimator undefined
        ph = yh.mean()
        p += w * ph
        var += w**2 * ph * (1 - ph) / max(len(yh) - 1, 1)
    return p, np.sqrt(var)


def main():
    pred = pd.read_parquet(PREP / "test_predictions.parquet")
    pred = pred.merge(pd.read_parquet(PREP / "test_predictions_F.parquet")[["test_id", "RF F"]], on="test_id")
    rows, agree = [], []
    for region in REGIONS:
        maps = {}
        for name in ("d_crop", "f_crop", "wc2020", "wc2021", "dea_mask"):
            a, tr = load(name, region)
            maps[name] = crop_mask(name, a)
        region_px = (load("wc2021", region)[0] != 255)
        for m in maps.values():
            m[~region_px] = 255
        pts = pred[pred.region == region]
        y = pts.label.to_numpy()
        n_crop_share = y.mean()
        se_srs = np.sqrt(n_crop_share * (1 - n_crop_share) / (len(y) - 1))
        area_km2 = region_px.sum() * 1e-4
        for name, m in maps.items():
            valid = m != 255
            W1 = float((m == 1).sum() / valid.sum())
            mp = at_points(m, tr, pts.lon, pts.lat)
            if name in FOREST:
                col, label, _ = FOREST[name]
                same = float((mp == pts[col].to_numpy()).mean())
                log.info("%s: %s map reproduces its test predictions at %.1f %% of %d points", region, label,
                         100 * same, len(pts))
            p, se = stratified({0: 1 - W1, 1: W1}, y, mp)
            rows.append(dict(region=region, map=name, region_km2=area_km2, n_points=len(y),
                             mapped_share=W1, mapped_km2=W1 * area_km2, nodata_share=float(1 - valid.sum() / region_px.sum()),
                             adjusted_share=p, adjusted_lo=p - 1.96 * se, adjusted_hi=p + 1.96 * se,
                             adjusted_km2=p * area_km2, points_share=n_crop_share,
                             points_lo=n_crop_share - 1.96 * se_srs, points_hi=n_crop_share + 1.96 * se_srs,
                             UA_crop=float(y[mp == 1].mean()) if (mp == 1).any() else np.nan,
                             PA_crop=float((mp[y == 1] == 1).mean())))
        w = maps["wc2021"]
        with rasterio.open(MAPS / f"wc2021_{region}.tif") as s:
            prof = s.profile
        for name, (_, label, key) in FOREST.items():
            d = maps[name]
            both = (d != 255) & (w != 255)
            cls = np.full(d.shape, 255, "uint8")
            cls[both] = d[both] + 2 * w[both]
            with rasterio.open(MAPS / f"{key}_vs_wc_{region}.tif", "w", **prof) as o:
                o.write(cls, 1)
            n = both.sum()
            agree.append(dict(region=region, model=label, neither=(cls == 0).sum() / n, d_only=(cls == 1).sum() / n,
                              wc_only=(cls == 2).sum() / n, both=(cls == 3).sum() / n,
                              d_only_km2=(cls == 1).sum() * 1e-4, wc_only_km2=(cls == 2).sum() * 1e-4))
            quicklook(region, d, cls, label, "" if key == "d" else f"_{key}")
    df = pd.DataFrame(rows)
    df.to_csv(CHECKS / "areas.csv", index=False)
    pd.DataFrame(agree).to_csv(CHECKS / "d_vs_wc.csv", index=False)
    log.info("\n%s", df[["region", "map", "mapped_share", "adjusted_share", "adjusted_lo", "adjusted_hi",
                         "points_share", "points_lo", "points_hi"]].round(3).to_string(index=False))
    log.info("\n%s", pd.DataFrame(agree).round(3).to_string(index=False))


def quicklook(region, d, cls, label="set D", suffix="", step=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    dd, cc = d[::step, ::step].astype(float), cls[::step, ::step].astype(float)
    dd[dd == 255] = np.nan
    cc[cc == 255] = np.nan
    fig, ax = plt.subplots(1, 2, figsize=(11, 6.5))
    ax[0].imshow(dd, cmap=ListedColormap(["#e9e4d4", "#c8862a"]), interpolation="nearest")
    ax[0].set_title(f"{region}: cropland, {label} forest")
    ax[1].imshow(cc, cmap=ListedColormap(["#e9e4d4", "#d1495b", "#3c6e9f", "#c8862a"]),
                 vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax[1].set_title("against WorldCover 2021")
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(color=c, label=name) for c, name in
                          (("#c8862a", "crop in both"), ("#d1495b", f"only {label} (WorldCover misses)"),
                           ("#3c6e9f", "only WorldCover"), ("#e9e4d4", "non-crop in both"))],
                 loc="lower left", fontsize=8, framealpha=0.9)
    for a in ax:
        a.set_axis_off()
    fig.tight_layout()
    fig.savefig(CHECKS / f"map_{region}{suffix}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
