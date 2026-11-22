"""Existing cropland maps at the samples and over the two mapped regions.

Maps:
  ESA WorldCover 2020 (v100) and 2021 (v200), 10 m, cropland = class 40
      (Microsoft Planetary Computer, collection esa-worldcover)
  Digital Earth Africa crop mask 2019, 10 m, EPSG:6933
      (explorer.digitalearth.africa STAC, collection crop_mask; public S3 bucket)
      bands: mask (0/1), prob (0-100), filtered (mask after object-based filtering)

Outputs:
  prep/benchmarks.parquet   sid, wc2020, wc2021, dea_mask, dea_prob, dea_filtered
  prep/maps/<map>_<region>.tif   region clips on the common 10 m grid (EPSG:32631), 255 = outside
  prep/checks/benchmark_scores.csv, prep/checks/benchmark_areas.csv

Usage: python s05_benchmarks.py points|regions
"""
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.features import geometry_mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from common import GDAL_ENV, GRID_CRS, GRID_RES, PREP, log_to, read_points, rowcol, signed, stac_client, to_crs

DEA_STAC = "https://explorer.digitalearth.africa/stac/search"
DEA_S3 = "https://deafrica-services.s3.af-south-1.amazonaws.com/"
CHECKS = PREP / "checks"
MAPS = PREP / "maps"
MAPS.mkdir(exist_ok=True)
log = log_to("benchmarks")


def sources():
    """(name, band, list of hrefs) for every map layer."""
    out = []
    pc = stac_client()
    for year in (2020, 2021):
        items = pc.search(collections=["esa-worldcover"], bbox=[-0.3, 5.9, 2.0, 11.3], datetime=str(year)).item_collection()
        out.append((f"wc{year}", [it.assets["map"].href.split("?")[0] for it in items]))
    feats = requests.get(DEA_STAC, params=dict(collections="crop_mask", bbox="-0.3,5.9,2.0,11.3", limit=200),
                         timeout=60).json()["features"]
    for band in ("mask", "prob", "filtered"):
        out.append((f"dea_{band}", [f["assets"][band]["href"].replace("s3://deafrica-services/", DEA_S3) for f in feats]))
    return out


def href(h):
    return signed(h) if "blob.core.windows.net" in h else h


def points():
    smp = pd.read_parquet(PREP / "samples.parquet")[["sid", "lon", "lat"]]
    res = pd.DataFrame({"sid": smp.sid})
    with rasterio.Env(**GDAL_ENV):
        for name, hrefs in sources():
            v = np.full(len(smp), np.nan, "float32")
            for h in hrefs:
                with rasterio.open(href(h)) as s:
                    x, y = to_crs(smp.lon, smp.lat, "EPSG:4326", s.crs)
                    r, c = rowcol(s, x, y)
                    ins = (r >= 0) & (r < s.height) & (c >= 0) & (c < s.width) & np.isnan(v)
                    if ins.any():
                        v[ins] = read_points(s, r[ins], c[ins])[:, 0, 0, 0]
            res[name] = v
            log.info(f"{name}: {np.isfinite(v).sum()} of {len(v)} samples have a value")
    res.to_parquet(PREP / "benchmarks.parquet", index=False)
    score_test(res)


def scores(y, p):
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    ua = tp / max(tp + fp, 1)
    pa = tp / max(tp + fn, 1)
    return dict(n=len(y), OA=(tp + tn) / len(y), UA_crop=ua, PA_crop=pa,
                F1_crop=2 * ua * pa / max(ua + pa, 1e-9), tp=tp, fp=fp, fn=fn, tn=tn)


def score_test(b):
    smp = pd.read_parquet(PREP / "samples.parquet")
    t = smp[smp.kind == "test"].merge(b, on="sid")
    # The DEA crop mask carries 255 inside its own tiles where it masks pixels out (mostly
    # water, tree cover and grassland in WorldCover). The map does not call them cropland,
    # so they count as non-crop. Five test points are affected, all labelled non-crop.
    for c in ("dea_mask", "dea_filtered"):
        t[c] = t[c].fillna(0)
    preds = {"WorldCover 2020": t.wc2020 == 40, "WorldCover 2021": t.wc2021 == 40,
             "DEA crop mask 2019": t.dea_mask == 1, "DEA crop mask 2019 (filtered)": t.dea_filtered == 1}
    rows = []
    for name, p in preds.items():
        for subset, sel in (("all", t.label.notna()), ("unanimous", t.agreement == "unanimous"),
                            ("split", t.agreement == "split")):
            rows.append(dict(map=name, subset=subset, **scores(t.label[sel].to_numpy(), p[sel].astype(int).to_numpy())))
    df = pd.DataFrame(rows)
    df.to_csv(CHECKS / "benchmark_scores.csv", index=False)
    print(df.round(3).to_string(index=False))


def regions():
    reg = gpd.read_file(PREP / "regions.gpkg", layer="regions").to_crs(GRID_CRS)
    reg = reg[reg.mapped]
    rows = []
    with rasterio.Env(**GDAL_ENV):
        for name, hrefs in sources():
            if name == "dea_prob":
                continue
            for r in reg.itertuples():
                x0, y0, x1, y1 = (np.array(r.geometry.bounds) // GRID_RES * GRID_RES) + [0, 0, GRID_RES, GRID_RES]
                w, h = int((x1 - x0) / GRID_RES), int((y1 - y0) / GRID_RES)
                tr = rasterio.transform.from_origin(x0, y1, GRID_RES, GRID_RES)
                out = np.full((h, w), 255, "uint8")
                bbox_ll = to_crs([x0, x1], [y0, y1], GRID_CRS, "EPSG:4326")
                for hr in hrefs:
                    with rasterio.open(href(hr)) as s:
                        bl = rasterio.warp.transform_bounds(s.crs, "EPSG:4326", *s.bounds)
                        if bl[0] > max(bbox_ll[0]) or bl[2] < min(bbox_ll[0]) or bl[1] > max(bbox_ll[1]) or bl[3] < min(bbox_ll[1]):
                            continue
                        # nearest neighbour onto the common grid: class codes must not be blended
                        with WarpedVRT(s, crs=GRID_CRS, transform=tr, width=w, height=h,
                                       resampling=Resampling.nearest, nodata=255) as v:
                            a = v.read(1)
                    fill = (out == 255) & (a != 255)
                    out[fill] = a[fill]
                inside = geometry_mask([r.geometry], (h, w), tr, invert=True)
                out[~inside] = 255
                crop = (out == 40) if name.startswith("wc") else (out == 1)
                f = MAPS / f"{name}_{r.region}.tif"
                with rasterio.open(f, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint8", crs=GRID_CRS,
                                   transform=tr, nodata=255, compress="deflate", tiled=True, predictor=2) as d:
                    d.write(out, 1)
                rows.append(dict(map=name, region=r.region, region_km2=inside.sum() * GRID_RES**2 / 1e6,
                                 crop_km2=crop.sum() * GRID_RES**2 / 1e6,
                                 nodata_share=float(((out == 255) & inside).sum() / inside.sum())))
                log.info(rows[-1])
    df = pd.DataFrame(rows)
    df["crop_share"] = df.crop_km2 / df.region_km2
    df.to_csv(CHECKS / "benchmark_areas.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    {"points": points, "regions": regions, "score": lambda: score_test(pd.read_parquet(PREP / "benchmarks.parquet"))}[sys.argv[1]]()
