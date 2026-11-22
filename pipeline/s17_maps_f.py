"""Step 5 redone with set F: wall-to-wall 10 m cropland maps of Savanes and Plateaux from the
OlmoEarth forest.

Model: the set F Random Forest of s16 (same training rows, common setting, same seed, so the
same model). Grid, blocks and assembly as in s13, on the same 10 m EPSG:32631 grid.

The OlmoEarth store has one group per UTM zone, both with origin (0, 9 502 720) and 10 m
pixels, so the zone 31N group shares our grid pixel for pixel. Its shards are 2048 px and
its inner chunks 256 px; blocks of 1024 px on a lattice at x = 0, y = 0 (mod 10 240 m) read
whole inner chunks only. A pixel west of 0 deg is read from the zone 30N group first, at
the nearest pixel of its own grid (as in s04 for the samples); pixels still empty are tried
in the other zone. Nodata is -128.

Outputs per region in prep/maps/:
  f_crop_<region>.tif   uint8, 1 crop, 0 non-crop, 255 outside the region or no data
  f_prob_<region>.tif   uint8, forest crop probability in percent, 255 nodata
Finished blocks are kept in prep/maps/f_blocks/, so an interrupted run resumes.
Usage: python s17_maps_f.py [region ...] | python s17_maps_f.py time   (a few blocks, timed)
"""
import json
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from sklearn.ensemble import RandomForestClassifier

from common import FEAT, GRID_CRS, PREP, YEAR, log_to, retry, to_crs
from s04_embeddings import dequantise_aef, olmoearth_root
from s09_tune import SEED, cap_rows, training_table
from s13_maps import BLK, LAT, MAPS, assemble, lattice_blocks

log = log_to("s17_maps_f")
warnings.filterwarnings("ignore", message="X does not have valid feature names")
AHEAD = 3  # blocks fetched ahead of the forest; each holds 128 MB of int8


def model() -> RandomForestClassifier:
    meta, _ = training_table()  # the same rows as sets A to E and s16
    cols = json.loads((FEAT / "columns.json").read_text())["F"]
    X = pd.read_parquet(FEAT / "set_F.parquet").set_index("sid").loc[meta.index, cols]
    cfg = json.loads((PREP / "checks/tune_best.json").read_text())["common"]
    rows = cap_rows(meta, int(cfg["cap"]), SEED)
    mf = cfg["max_features"]
    rf = RandomForestClassifier(300, max_features=float(mf) if mf != "sqrt" else mf,
                                min_samples_leaf=int(cfg["min_samples_leaf"]), n_jobs=-1, random_state=SEED)
    return rf.fit(X.loc[rows], meta.label.loc[rows])


class Store:
    def __init__(self):
        root = olmoearth_root()
        self.z = {zn: root[zn] for zn in ("utm30", "utm31")}
        self.t = {zn: int(np.flatnonzero(g["time"][:] == YEAR)[0]) for zn, g in self.z.items()}

    def values(self, zn: str, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """int8 (128, n) at grid pixel centres x, y (EPSG:32631), nearest pixel of zone zn."""
        g = self.z[zn]
        if zn != "utm31":
            x, y = to_crs(x, y, GRID_CRS, g.attrs["proj:code"])
        a, _, c, _, e, f = g.attrs["spatial:transform"]
        col = np.floor((x - c) / a).astype(int)
        row = np.floor((y - f) / e).astype(int)
        r0, r1, c0, c1 = row.min(), row.max() + 1, col.min(), col.max() + 1
        w = retry(lambda: g["embeddings"][self.t[zn], :, r0:r1, c0:c1])
        return w[:, row - r0, col - c0]

    def block(self, x0: float, y1: float) -> np.ndarray:
        """int8 (128, BLK * BLK) for the block with top-left corner (x0, y1), row-major."""
        xs = x0 + 5 + 10 * np.arange(BLK)
        ys = y1 - 5 - 10 * np.arange(BLK)
        lon_w, _ = to_crs([x0, x0], [y1, y1 - LAT], GRID_CRS, "EPSG:4326")
        if min(lon_w) >= 0:  # whole block east of 0 deg: one aligned zone 31N window
            r0 = round((self.z["utm31"].attrs["spatial:transform"][5] - y1) / 10)
            c0 = round(x0 / 10)
            w = retry(lambda: self.z["utm31"]["embeddings"][self.t["utm31"], :, r0:r0 + BLK, c0:c0 + BLK])
            out = np.asarray(w).reshape(128, -1)
            first = None
        else:
            X, Y = (v.ravel() for v in np.meshgrid(xs, ys))
            lon, _ = to_crs(X, Y, GRID_CRS, "EPSG:4326")
            first = np.where(lon < 0, "utm30", "utm31")
            out = np.full((128, BLK * BLK), -128, "int8")
        if first is not None:
            for zones in (first, np.where(first == "utm30", "utm31", "utm30")):
                for zn in ("utm30", "utm31"):
                    idx = np.flatnonzero((zones == zn) & (out[0] == -128))
                    if len(idx):
                        v = self.values(zn, X[idx], Y[idx])
                        got = v[0] != -128
                        out[:, idx[got]] = v[:, got]
        return out


def run_region(name: str, rf, store: Store, limit: int | None = None):
    reg = gpd.read_file(PREP / "regions.gpkg", layer="regions").to_crs(GRID_CRS)
    geom = reg.loc[reg.region == name, "geometry"].iloc[0]
    bdir = MAPS / "f_blocks" / name
    bdir.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in lattice_blocks(name, 0.0, 0.0) if not (bdir / f"{int(j[0])}_{int(j[1])}.npy").exists()]
    if limit:
        jobs = jobs[:limit]
    log.info("%s: %d blocks to do", name, len(jobs))
    t0 = time.time()
    with ThreadPoolExecutor(AHEAD) as ex:
        futs = [ex.submit(store.block, *j) for j in jobs[:AHEAD]]
        for k, (x, y) in enumerate(jobs):
            q = futs[k].result()
            futs[k] = None
            if k + AHEAD < len(jobs):
                futs.append(ex.submit(store.block, *jobs[k + AHEAD]))
            btr = rasterio.transform.from_origin(x, y, 10, 10)
            m = (geometry_mask([geom], (BLK, BLK), btr, invert=True).ravel() & (q[0] != -128))
            prob = np.full(BLK * BLK, 255, "uint8")
            idx = np.flatnonzero(m)
            for s in range(0, len(idx), 262_144):  # 128 MB of float32 at a time
                i = idx[s:s + 262_144]
                prob[i] = np.rint(rf.predict_proba(dequantise_aef(q[:, i]).T)[:, 1] * 100).astype("uint8")
            del q
            np.save(bdir / f"{int(x)}_{int(y)}.npy", prob.reshape(BLK, BLK))
            if k % 10 == 0:
                log.info("%s block %d/%d, %.0f s", name, k + 1, len(jobs), time.time() - t0)
    return time.time() - t0, len(jobs)


def main(args):
    store = Store()
    rf = model()
    if args == ["time"]:
        sec, n = run_region("Savanes", rf, store, limit=4)
        log.info("timing: %d blocks in %.0f s", n, sec)
        return
    for name in args or ["Savanes", "Plateaux"]:
        run_region(name, rf, store)
        assemble(name, "f")


if __name__ == "__main__":
    main(sys.argv[1:])
