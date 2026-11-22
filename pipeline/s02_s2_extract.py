"""Sentinel-2 L2A time series of 2019 at every sample location.

Source: Microsoft Planetary Computer, collection sentinel-2-l2a (no login).
All 2019 items over Togo are processing baseline 02.12, so reflectance = DN / 10000
with no BOA offset (the offset only starts with baseline 04.00 in 2022).

Each sample gets one home MGRS tile, the tile in which it lies furthest from the edge,
so overlapping tiles are not read twice. For every item of a home tile the scene
classification (SCL, 20 m, under 1 MB) is read whole; the ten bands are then read only
for samples whose SCL code is in SCL_READ, one window per 512 px internal block.

Outputs:
  prep/s2/catalog.parquet          one row per item (tile, date, cloud cover, href per asset)
  prep/s2/home_tile.parquet        sid -> home tile
  prep/s2/items/<item id>.parquet  sid, scl, B02 ... B12 (uint16 DN, 0 = not read)

Rerunning skips finished items, so an interrupted run can simply be restarted.
Usage: python s02_s2_extract.py [--limit N] [--workers 12]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import rasterio

from common import (GDAL_ENV, PREP, S2_BANDS, SCL_READ, TOGO_BBOX, YEAR, log_to,
                    read_points, retry, rowcol, signed, stac_client, to_crs)

OUT = PREP / "s2"
ITEMS = OUT / "items"
ITEMS.mkdir(parents=True, exist_ok=True)
log = log_to("s2_extract")


def build_catalog() -> pd.DataFrame:
    f = OUT / "catalog.parquet"
    if f.exists():
        return pd.read_parquet(f)
    items = list(stac_client().search(collections=["sentinel-2-l2a"], bbox=list(TOGO_BBOX),
                                      datetime=f"{YEAR}-01-01/{YEAR}-12-31").items())
    rows = []
    for it in items:
        p = it.properties
        b10 = it.assets["B02"].extra_fields
        rows.append(dict(
            item=it.id, tile=p["s2:mgrs_tile"], datetime=pd.Timestamp(it.datetime).tz_convert(None),
            platform=p["platform"], rel_orbit=p.get("sat:relative_orbit"), cloud=p["eo:cloud_cover"],
            baseline=p.get("s2:processing_baseline"), epsg=int(str(p.get("proj:code", p.get("proj:epsg"))).split(":")[-1]),
            transform10=json.dumps(b10.get("proj:transform")), shape10=json.dumps(b10.get("proj:shape")),
            hrefs=json.dumps({a: it.assets[a].href.split("?")[0] for a in S2_BANDS + ["SCL"]}),
        ))
    cat = pd.DataFrame(rows).sort_values(["tile", "datetime"]).reset_index(drop=True)
    cat.to_parquet(f, index=False)
    return cat


def assign_home_tiles(cat: pd.DataFrame, smp: pd.DataFrame) -> pd.DataFrame:
    f = OUT / "home_tile.parquet"
    if f.exists():
        return pd.read_parquet(f)
    tiles = cat.drop_duplicates("tile").set_index("tile")
    best = np.full(len(smp), -np.inf)
    home = np.full(len(smp), None, dtype=object)
    for tile, r in tiles.iterrows():
        t = json.loads(r.transform10)
        shp = json.loads(r.shape10)
        x0, y1 = t[2], t[5]
        x1, y0 = x0 + t[0] * shp[1], y1 + t[4] * shp[0]
        x, y = to_crs(smp.lon, smp.lat, "EPSG:4326", f"EPSG:{r.epsg}")
        margin = np.minimum.reduce([x - x0, x1 - x, y - y0, y1 - y])
        better = margin > best
        best[better], home[better] = margin[better], tile
    out = pd.DataFrame(dict(sid=smp.sid, tile=home, margin_m=best))
    if (out.margin_m <= 0).any():
        raise RuntimeError(f"{(out.margin_m <= 0).sum()} samples outside every tile")
    out.to_parquet(f, index=False)
    return out


def extract_item(row, pts: pd.DataFrame) -> str:
    dst = ITEMS / f"{row.item}.parquet"
    if dst.exists():
        return "skip"
    hrefs = json.loads(row.hrefs)
    x, y = to_crs(pts.lon, pts.lat, "EPSG:4326", f"EPSG:{row.epsg}")
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(signed(hrefs["SCL"])) as s:
            r, c = rowcol(s, x, y)
            scl_full = s.read(1)
        ok = (r >= 0) & (r < scl_full.shape[0]) & (c >= 0) & (c < scl_full.shape[1])
        scl = np.zeros(len(pts), "uint8")
        scl[ok] = scl_full[r[ok], c[ok]]
        del scl_full
        seen = scl > 0
        read = np.isin(scl, SCL_READ)
        vals = np.zeros((len(pts), len(S2_BANDS)), "uint16")
        if read.any():
            for k, b in enumerate(S2_BANDS):
                with rasterio.open(signed(hrefs[b])) as s:
                    r, c = rowcol(s, x[read], y[read])
                    v = read_points(s, r, c)[:, 0, 0, 0]
                vals[read, k] = np.nan_to_num(v, nan=0).astype("uint16")
    df = pd.DataFrame(vals[seen], columns=S2_BANDS)
    df.insert(0, "scl", scl[seen])
    df.insert(0, "sid", pts.sid.to_numpy()[seen].astype("int32"))
    tmp = dst.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.rename(dst)
    return f"{seen.sum()} seen, {read.sum()} read"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    smp = pd.read_parquet(PREP / "samples.parquet")
    cat = build_catalog()
    home = assign_home_tiles(cat, smp)
    smp = smp.merge(home, on="sid")
    todo = cat[cat.tile.isin(set(home.tile))]
    log.info(f"{len(cat)} items, {cat.tile.nunique()} tiles; {len(todo)} items on {home.tile.nunique()} home tiles")
    if a.limit:
        todo = todo.sample(a.limit, random_state=0)
    groups = {t: g[["sid", "lon", "lat"]].reset_index(drop=True) for t, g in smp.groupby("tile")}
    t0, done, failed = time.time(), 0, []
    with ThreadPoolExecutor(a.workers) as ex:
        fut = {ex.submit(retry, lambda r=r: extract_item(r, groups[r.tile])): r.item for r in todo.itertuples()}
        for f in as_completed(fut):
            done += 1
            try:
                msg = f.result()
            except Exception as e:  # keep going; failed items are retried on the next run
                failed.append(fut[f])
                msg = f"FAILED {e!r}"[:300]
            if msg != "skip" or done % 200 == 0:
                rate = done / (time.time() - t0)
                log.info(f"{done}/{len(todo)} {fut[f]}: {msg} ({rate * 60:.1f} items/min)")
    log.info(f"finished in {(time.time() - t0) / 60:.1f} min, {len(failed)} failed: {failed[:10]}")


if __name__ == "__main__":
    main()
