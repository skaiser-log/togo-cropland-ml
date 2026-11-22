"""Sentinel-1 RTC backscatter of 2019 at every sample location.

Source: Microsoft Planetary Computer, collection sentinel-1-rtc (radiometrically
terrain corrected gamma0, linear power, 10 m, VV and VH; no login needed to read).
All 2019 scenes over Togo are ascending passes on relative orbits 1, 74 and 147.

Per scene and polarisation the value stored is the mean of the 3 x 3 pixels around the
sample, computed in linear power, as a light speckle filter; the temporal aggregation in
the feature step reduces speckle further. Neighbouring frames of one pass overlap, so a
sample can appear twice for one date; the feature step drops those duplicates.

Outputs:
  prep/s1/catalog.parquet          one row per scene
  prep/s1/items/<item id>.parquet  sid, vv, vh (linear gamma0, float32), n_valid (of 9)

Usage: python s03_s1_extract.py [--limit N] [--workers 8]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio

from common import (GDAL_ENV, PREP, TOGO_BBOX, YEAR, log_to, read_points, retry, rowcol, signed,
                    stac_client, to_crs)

OUT = PREP / "s1"
ITEMS = OUT / "items"
ITEMS.mkdir(parents=True, exist_ok=True)
log = log_to("s1_extract")


def build_catalog() -> pd.DataFrame:
    f = OUT / "catalog.parquet"
    if f.exists():
        return pd.read_parquet(f)
    items = list(stac_client().search(collections=["sentinel-1-rtc"], bbox=list(TOGO_BBOX),
                                      datetime=f"{YEAR}-01-01/{YEAR}-12-31").items())
    rows = [dict(item=it.id, datetime=pd.Timestamp(it.datetime).tz_convert(None),
                 platform=it.properties["platform"], orbit_state=it.properties["sat:orbit_state"],
                 rel_orbit=it.properties["sat:relative_orbit"],
                 epsg=int(it.properties["proj:code"].split(":")[-1]),
                 bbox=json.dumps(it.properties["proj:bbox"]),
                 hrefs=json.dumps({a: it.assets[a].href.split("?")[0] for a in ("vv", "vh")}))
            for it in items]
    cat = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    cat.to_parquet(f, index=False)
    return cat


def extract_item(row, smp: pd.DataFrame) -> str:
    dst = ITEMS / f"{row.item}.parquet"
    if dst.exists():
        return "skip"
    x0, y0, x1, y1 = json.loads(row.bbox)
    x, y = to_crs(smp.lon, smp.lat, "EPSG:4326", f"EPSG:{row.epsg}")
    inb = (x > x0 + 15) & (x < x1 - 15) & (y > y0 + 15) & (y < y1 - 15)
    hrefs = json.loads(row.hrefs)
    out = {}
    with rasterio.Env(**GDAL_ENV):
        for pol in ("vv", "vh") if inb.any() else ():
            with rasterio.open(signed(hrefs[pol])) as s:
                r, c = rowcol(s, x[inb], y[inb])
                win = read_points(s, r, c, halo=1)[:, 0]  # (n, 3, 3)
            win[win <= 0] = np.nan  # zero is padding at the swath edge, not a backscatter value
            flat = win.reshape(len(win), -1)
            n = np.isfinite(flat).sum(axis=1)
            out[pol] = np.where(n > 0, np.nansum(flat, axis=1) / np.maximum(n, 1), np.nan)
            out[f"n_{pol}"] = n
    if not inb.any():
        out = {k: np.empty(0, "float32") for k in ("vv", "vh", "n_vv", "n_vh")}
    keep = np.isfinite(out["vv"]) & np.isfinite(out["vh"])
    df = pd.DataFrame(dict(sid=smp.sid.to_numpy()[inb][keep].astype("int32"),
                           vv=out["vv"][keep].astype("float32"), vh=out["vh"][keep].astype("float32"),
                           n_valid=np.minimum(out["n_vv"], out["n_vh"])[keep].astype("uint8")))
    tmp = dst.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(dst)
    return f"{inb.sum()} in bounds, {keep.sum()} valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    smp = pd.read_parquet(PREP / "samples.parquet")[["sid", "lon", "lat"]]
    cat = build_catalog()
    todo = cat.sample(a.limit, random_state=0) if a.limit else cat
    log.info(f"{len(cat)} scenes; processing {len(todo)}")
    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        fut = {ex.submit(retry, lambda r=r: extract_item(r, smp)): r.item for r in todo.itertuples()}
        for f in as_completed(fut):
            done += 1
            try:
                msg = f.result()
            except Exception as e:
                failed.append(fut[f])
                msg = f"FAILED {e!r}"[:300]
            if msg != "skip" or done % 100 == 0:
                log.info(f"{done}/{len(todo)} {fut[f]}: {msg} ({done / (time.time() - t0) * 60:.1f} scenes/min)")
    log.info(f"finished in {(time.time() - t0) / 60:.1f} min, {len(failed)} failed: {failed[:10]}")


if __name__ == "__main__":
    main()
