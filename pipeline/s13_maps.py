"""Step 5: wall-to-wall 10 m cropland maps of Savanes and Plateaux from the AlphaEarth forest.

Model: the set D Random Forest of s12 (same training rows, common setting, same seed, so
the same model). Grid: the 10 m EPSG:32631 grid of the benchmark clips in prep/maps.

The grid is processed in 1024 x 1024 px blocks aligned to the AlphaEarth tile lattice
(see LAT_X below) and pasted into the region grid. Each block takes its 64 AlphaEarth
bands from the tiles whose footprint touches it, warped onto the block with nearest
neighbour (zone 31N tiles share the grid; zone 30N tiles, west of 0 deg, are resampled by
under half a pixel). AlphaEarth nodata (-128) is left for the next tile, as in s04.
Finished blocks are kept as .npy files, so an interrupted run resumes.

Outputs per region in prep/maps/:
  d_crop_<region>.tif   uint8, 1 crop, 0 non-crop, 255 outside the region or no data
  d_prob_<region>.tif   uint8, forest crop probability in percent, 255 nodata
Usage: python s13_maps.py [region ...] | python s13_maps.py time   (one block, timed)
"""
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from shapely.geometry import box
from sklearn.ensemble import RandomForestClassifier

from common import GDAL_ENV, GRID_CRS, PREP, log_to
from s04_embeddings import aef_tiles, dequantise_aef
from s09_tune import SEED, cap_rows, training_table

log = log_to("s13_maps")
warnings.filterwarnings("ignore", message="X does not have valid feature names")
MAPS = PREP / "maps"
BLK = 1024


def model() -> RandomForestClassifier:
    meta, X = training_table()
    cfg = json.loads((PREP / "checks/tune_best.json").read_text())["common"]
    rows = cap_rows(meta, int(cfg["cap"]), SEED)
    mf = cfg["max_features"]
    rf = RandomForestClassifier(300, max_features=float(mf) if mf != "sqrt" else mf,
                                min_samples_leaf=int(cfg["min_samples_leaf"]), n_jobs=-1, random_state=SEED)
    return rf.fit(X["D"].loc[rows], meta.label.loc[rows])


def region_grid(name: str):
    """Grid of the benchmark clip for this region, and the region mask on it."""
    with rasterio.open(MAPS / f"wc2021_{name}.tif") as s:
        tr, w, h = s.transform, s.width, s.height
    reg = gpd.read_file(PREP / "regions.gpkg", layer="regions").to_crs(GRID_CRS)
    geom = reg.loc[reg.region == name, "geometry"].iloc[0]
    inside = geometry_mask([geom], (h, w), tr, invert=True)
    return tr, w, h, inside


# AlphaEarth tiles of one UTM zone sit on a common lattice: edges at x = 8480 and y = 0
# (mod 10 240 m), and their internal blocks are 1024 px = 10 240 m. Processing blocks on
# that lattice read exactly one internal block per band from a zone 31N tile; blocks offset
# from it read up to four and ran three times slower.
LAT_X, LAT_Y, LAT = 8480.0, 0.0, BLK * 10.0


def read_block(tiles: gpd.GeoDataFrame, btr) -> np.ndarray:
    x0, y1 = btr * (0, 0)
    x1, y0 = btr * (BLK, BLK)
    bb = box(*transform_bounds(GRID_CRS, "EPSG:4326", x0, y0, x1, y1))
    out = np.full((64, BLK, BLK), -128, "int8")
    for t in tiles[tiles.intersects(bb)].itertuples():
        with rasterio.Env(**GDAL_ENV), rasterio.open(t.url) as s, \
                WarpedVRT(s, crs=GRID_CRS, transform=btr, width=BLK, height=BLK,
                          resampling=Resampling.nearest, nodata=-128) as v:
            a = v.read()
        fill = (out[0] == -128) & (a[0] != -128)
        out[:, fill] = a[:, fill]
    return out


def lattice_blocks(name: str, lat_x: float = LAT_X, lat_y: float = LAT_Y) -> list[tuple[float, float]]:
    """Top-left corners (x, y) of lattice blocks that touch the region."""
    tr, w, h, _ = region_grid(name)
    X0, Y1 = tr * (0, 0)
    X1, Y0 = tr * (w, h)
    xs = np.arange(np.floor((X0 - lat_x) / LAT) * LAT + lat_x, X1, LAT)
    ys = np.arange(np.ceil((Y1 - lat_y) / LAT) * LAT + lat_y, Y0, -LAT)
    reg = gpd.read_file(PREP / "regions.gpkg", layer="regions").to_crs(GRID_CRS)
    geom = reg.loc[reg.region == name, "geometry"].iloc[0]
    return [(x, y) for y in ys for x in xs if geom.intersects(box(x, y - LAT, x + LAT, y))]


def run_region(name: str, rf, tiles, limit: int | None = None):
    reg = gpd.read_file(PREP / "regions.gpkg", layer="regions").to_crs(GRID_CRS)
    geom = reg.loc[reg.region == name, "geometry"].iloc[0]
    bdir = MAPS / "d_blocks" / name
    bdir.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in lattice_blocks(name) if not (bdir / f"{int(j[0])}_{int(j[1])}.npy").exists()]
    if limit:
        jobs = jobs[:limit]
    log.info("%s: %d blocks to do", name, len(jobs))

    def fetch(job):
        btr = rasterio.transform.from_origin(job[0], job[1], 10, 10)
        return job, btr, read_block(tiles, btr)

    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for k, ((x, y), btr, q) in enumerate(ex.map(fetch, jobs)):
            m = geometry_mask([geom], (BLK, BLK), btr, invert=True) & (q[0] != -128)
            prob = np.full((BLK, BLK), 255, "uint8")
            if m.any():
                prob[m] = np.rint(rf.predict_proba(dequantise_aef(q[:, m]).T)[:, 1] * 100).astype("uint8")
            np.save(bdir / f"{int(x)}_{int(y)}.npy", prob)
            if k % 10 == 0:
                log.info("%s block %d/%d, %.0f s", name, k + 1, len(jobs), time.time() - t0)
    return time.time() - t0, len(jobs)


def assemble(name: str, prefix: str = "d"):
    tr, w, h, inside = region_grid(name)
    prob = np.full((h, w), 255, "uint8")
    X0, Y1 = tr * (0, 0)
    for f in (MAPS / f"{prefix}_blocks" / name).glob("*.npy"):
        x, y = map(float, f.stem.split("_"))
        b = np.load(f)
        c0, r0 = round((x - X0) / 10), round((Y1 - y) / 10)
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + BLK, h), min(c0 + BLK, w)
        if rs < re and cs < ce:
            part = b[rs - r0:re - r0, cs - c0:ce - c0]
            dst = prob[rs:re, cs:ce]
            dst[part != 255] = part[part != 255]
    crop = np.where(prob == 255, 255, (prob >= 50).astype("uint8"))
    prof = dict(driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs=GRID_CRS, transform=tr,
                nodata=255, compress="deflate", tiled=True, predictor=2)
    for arr, stem in ((crop, f"{prefix}_crop"), (prob, f"{prefix}_prob")):
        with rasterio.open(MAPS / f"{stem}_{name}.tif", "w", **prof) as d:
            d.write(arr, 1)
    nod = float(((prob == 255) & inside).sum() / inside.sum())
    log.info("%s %s assembled: crop share %.3f, nodata share inside region %.4f", name, prefix,
             float((crop == 1).sum() / ((crop != 255) & inside).sum()), nod)


def main(args):
    tiles = aef_tiles()
    rf = model()
    if args == ["time"]:
        sec, n = run_region("Savanes", rf, tiles, limit=12)
        log.info("timing: %d blocks in %.0f s", n, sec)
        return
    for name in args or ["Savanes", "Plateaux"]:
        run_region(name, rf, tiles)
        assemble(name)


if __name__ == "__main__":
    main(sys.argv[1:])
