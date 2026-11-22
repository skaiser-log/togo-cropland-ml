"""Foundation-model embeddings of 2019 at every sample location (feature sets D and E).

AlphaEarth Foundations (Google, Google DeepMind; CC BY 4.0), annual 2019, via Source
Cooperative (tge-labs/aef/v1/annual). 64 int8 bands per 10 m pixel, nodata -128.
The tiles are stored bottom-up (positive y pixel size). rasterio's affine maths handles
that, and ``orientation_check`` proves it before any sampling: on a window over Lake
Nangbeto k-means clusters of the embedding must match the WorldCover water mask
at the stated coordinates, and clearly better than with the window flipped vertically.

TESSERA v1.1 (University of Cambridge), 2019, 128 float32 dimensions per 10 m pixel,
read point-wise from the public zarr store with geotessera.

Both sources are read from the Source Cooperative S3 bucket directly, not through the
data.source.coop gateway.

Outputs: prep/aef_2019.parquet (sid, tile, a00..a63 raw int8),
         prep/tessera_2019.parquet (sid, t000..t127 float32),
         prep/checks/aef_orientation.json
OlmoEarth v1.3 (Ai2, OlmoEarth Artifact License), annual 2019, 128 int8 dimensions from
Sentinel-1, Sentinel-2 and Landsat 8/9, Zarr v3 on Source Cooperative; same dequantisation
as AlphaEarth. Output prep/olmoearth_2019.parquet (sid, zone, o000..o127 raw int8).

Usage: python s04_embeddings.py aef|tessera|olmoearth|check
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from common import GDAL_ENV, PREP, RAW, YEAR, log_to, read_points, retry, rowcol, stac_client, to_crs

S3 = "https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/"
CHECKS = PREP / "checks"
CHECKS.mkdir(exist_ok=True)
log = log_to("embeddings")


def aef_tiles() -> gpd.GeoDataFrame:
    g = gpd.read_file(RAW / "aef_index.gpkg", bbox=(-0.3, 5.9, 2.0, 11.3))
    g = g[g.year == YEAR].copy()
    g["url"] = "/vsicurl/" + g.path.str.replace("s3://us-west-2.opendata.source.coop/", S3, regex=False)
    return g.reset_index(drop=True)


def dequantise_aef(q: np.ndarray) -> np.ndarray:
    """AlphaEarth int8 -> float: (q / 127.5)^2 * sign(q); -128 is nodata."""
    q = q.astype("float32")
    out = np.sign(q) * (q / 127.5) ** 2
    out[q == -128] = np.nan
    return out


def orientation_check():
    """Lake Nangbeto: k-means cluster purity against WorldCover water, as stored and flipped."""
    from sklearn.cluster import KMeans

    lon, lat, half = 1.43, 7.45, 3000  # 6 km window over the reservoir and its shore
    tiles = aef_tiles()
    t = tiles[tiles.geometry.contains(gpd.points_from_xy([lon], [lat])[0])].iloc[0]
    with rasterio.Env(**GDAL_ENV), rasterio.open(t.url) as s:
        x, y = to_crs([lon], [lat], "EPSG:4326", s.crs)
        r, c = rowcol(s, x, y)
        n = int(half / 10)
        w = Window(int(c[0]) - n, int(r[0]) - n, 2 * n, 2 * n)
        q = s.read(window=w)
        tr = s.window_transform(w)
        emb = dequantise_aef(q).reshape(64, -1).T
    rr, cc = np.mgrid[0:2 * n, 0:2 * n]
    px, py = tr * (cc.ravel() + 0.5, rr.ravel() + 0.5)
    wc = stac_client().search(collections=["esa-worldcover"], intersects={"type": "Point", "coordinates": [lon, lat]},
                              datetime="2020").item_collection()[0]

    def water_at(xs, ys):
        with rasterio.Env(**GDAL_ENV), rasterio.open(wc.assets["map"].href) as s:
            lx, ly = to_crs(xs, ys, t.crs, s.crs)
            rws, cls = rowcol(s, lx, ly)
            v = read_points(s, rws, cls)[:, 0, 0, 0]
        return v == 80

    # k-means with several clusters (land is heterogeneous), each cluster mapped to its majority
    # WorldCover class; purity is the share of pixels that mapping gets right. The alternative
    # hypothesis is the same window flipped vertically: same water share, shoreline misplaced.
    ok = np.isfinite(emb).all(axis=1)
    lab = KMeans(8, n_init=4, random_state=0).fit_predict(emb[ok])
    wat = water_at(px, py).reshape(2 * n, 2 * n)

    def purity(w):
        w = w.ravel()[ok]
        return float(sum(max(w[lab == k].sum(), (~w[lab == k]).sum()) for k in range(8)) / len(w))

    res = {"water_share": round(float(wat.mean()), 3),
           "purity_as_stored": round(purity(wat), 3),
           "purity_flipped": round(purity(np.flipud(wat)), 3)}
    res["passed"] = bool(0.1 < res["water_share"] < 0.9 and res["purity_as_stored"] > 0.95
                         and res["purity_as_stored"] - res["purity_flipped"] > 0.1)
    res.update(tile=t.path, centre=[lon, lat], window_px=2 * n)
    (CHECKS / "aef_orientation.json").write_text(json.dumps(res, indent=2))
    log.info(f"AEF orientation check: {res}")
    if not res["passed"]:
        raise SystemExit("AlphaEarth orientation check failed; do not sample")


def extract_aef():
    if not json.loads((CHECKS / "aef_orientation.json").read_text())["passed"]:
        raise SystemExit("run the orientation check first")
    smp = pd.read_parquet(PREP / "samples.parquet")[["sid", "lon", "lat"]]
    tiles = aef_tiles()
    # A tile is a UTM rectangle, but it only holds data inside its own zone: zone 30N tiles
    # reach east of 0 deg as empty raster. Candidates are therefore the tiles whose footprint
    # (index geometry) contains the sample, ordered by distance from the rectangle edge; a
    # sample still empty after its first candidate is retried on the next one.
    pts = gpd.GeoSeries(gpd.points_from_xy(smp.lon, smp.lat), crs=4326)
    margin = np.full((len(smp), len(tiles)), -np.inf)
    for i, t in tiles.iterrows():
        x, y = to_crs(smp.lon, smp.lat, "EPSG:4326", t.crs)
        m = np.minimum.reduce([x - t.utm_west, t.utm_east - x, y - t.utm_south, t.utm_north - y])
        inside = pts.within(t.geometry.buffer(1e-4)).to_numpy() & (m > 0)
        margin[inside, i] = m[inside]
    order = np.argsort(-margin, axis=1)
    assert np.isfinite(margin.max(axis=1)).all(), "samples outside every AlphaEarth tile"

    def one(i, sel):
        t = tiles.loc[i]
        x, y = to_crs(smp.lon.to_numpy()[sel], smp.lat.to_numpy()[sel], "EPSG:4326", t.crs)

        def go():
            with rasterio.Env(**GDAL_ENV), rasterio.open(t.url) as s:
                r, c = rowcol(s, x, y)
                return read_points(s, r, c)[:, :, 0, 0]
        v = retry(go)
        log.info(f"AEF tile {i}: {len(sel)} samples")
        return sel, v

    q = np.full((len(smp), 64), -128, "int8")
    used = np.full(len(smp), -1)
    for k in range(3):  # first, second and third candidate tile
        todo = np.flatnonzero((q == -128).all(axis=1) & np.isfinite(margin[np.arange(len(smp)), order[:, k]]))
        if not len(todo):
            break
        cand = order[todo, k]
        with ThreadPoolExecutor(8) as ex:
            for sel, v in ex.map(lambda i: one(i, todo[cand == i]), sorted(set(cand))):
                got = ~np.isnan(v).all(axis=1)
                q[sel[got]] = np.nan_to_num(v[got], nan=-128).astype("int8")
                used[sel[got]] = cand[np.searchsorted(todo, sel[got])]
        log.info(f"AEF pass {k + 1}: {len(todo)} samples tried, still empty {(q == -128).all(axis=1).sum()}")
    df = pd.DataFrame(q, columns=[f"a{k:02d}" for k in range(64)])
    df.insert(0, "tile", np.where(used >= 0, tiles.path.str.split("/").str[-1].to_numpy()[np.maximum(used, 0)], None))
    df.insert(0, "sid", smp.sid.to_numpy())
    df.to_parquet(PREP / "aef_2019.parquet", index=False)
    log.info(f"AEF: {len(df)} samples, nodata rows {(q == -128).all(axis=1).sum()}")


def extract_tessera():
    from geotessera.store import GeoTesseraZarr

    smp = pd.read_parquet(PREP / "samples.parquet")[["sid", "lon", "lat"]]
    gt = GeoTesseraZarr(cache_dir=str(PREP / "tessera_cache"))
    log.info(f"TESSERA store {gt.url}, years {gt.years}")
    parts = []
    for k in range(0, len(smp), 5000):  # chunks, so progress shows and a failure costs little
        f = PREP / "tessera_parts" / f"{k:06d}.npy"
        f.parent.mkdir(exist_ok=True)
        if not f.exists():
            chunk = smp.iloc[k:k + 5000]
            v = retry(lambda: gt.sample_points(list(zip(chunk.lon, chunk.lat)), year=YEAR))
            np.save(f, v.astype("float32"))
            log.info(f"TESSERA {k + len(chunk)}/{len(smp)}, NaN rows {np.isnan(v).all(axis=1).sum()}")
        parts.append(np.load(f))
    v = np.concatenate(parts)
    df = pd.DataFrame(v, columns=[f"t{k:03d}" for k in range(v.shape[1])])
    df.insert(0, "sid", smp.sid.to_numpy())
    df.to_parquet(PREP / "tessera_2019.parquet", index=False)
    log.info(f"TESSERA: {len(df)} samples x {v.shape[1]}, NaN rows {np.isnan(v).all(axis=1).sum()}")


def olmoearth_root():
    """OlmoEarth v1.3 annual embeddings (Ai2, OlmoEarth Artifact License), Zarr v3 on Source
    Cooperative, read from the S3 bucket directly."""
    import zarr
    from obstore.store import S3Store

    zarr.config.set({"array.sharding_coalesce_max_gap_bytes": 0})
    s3 = S3Store("us-west-2.opendata.source.coop", prefix="olmoearth/olmoearth-v1-3-embeddings/embeddings.zarr",
                 region="us-west-2", skip_signature=True)
    return zarr.open_group(zarr.storage.ObjectStore(s3, read_only=True), mode="r")


def extract_olmoearth():
    """128 int8 dimensions per sample for 2019. Samples are grouped by the store's 256 x 256 px
    inner chunks; each group reads the smallest window around its points. A sample is read from
    the UTM zone of its longitude first, the neighbouring zone second."""
    root = olmoearth_root()
    smp = pd.read_parquet(PREP / "samples.parquet")[["sid", "lon", "lat"]]
    parts = PREP / "olmoearth_parts"
    parts.mkdir(exist_ok=True)
    q = np.full((len(smp), 128), -128, "int8")
    zone_used = np.full(len(smp), "", dtype=object)
    first = np.where(smp.lon < 0, "utm30", "utm31")
    for k, zones in enumerate((first, np.where(first == "utm30", "utm31", "utm30"))):
        for zn in ("utm30", "utm31"):
            sel = np.flatnonzero((zones == zn) & (q == -128).all(axis=1))
            f = parts / f"pass{k}_{zn}.npy"
            if len(sel) == 0:
                continue
            if f.exists():
                v = np.load(f)
            else:
                z = root[zn]
                arr = z["embeddings"]
                t = int(np.flatnonzero(z["time"][:] == YEAR)[0])
                x, y = to_crs(smp.lon.to_numpy()[sel], smp.lat.to_numpy()[sel], "EPSG:4326", z.attrs["proj:code"])
                a, _, c, _, e, fy = z.attrs["spatial:transform"]
                col = np.floor((x - c) / a).astype(int)
                row = np.floor((y - fy) / e).astype(int)
                key = (row // 256) * 10_000_000 + col // 256
                v = np.full((len(sel), 128), -128, "int8")
                groups = [np.flatnonzero(key == g) for g in np.unique(key)]
                log.info(f"OlmoEarth {zn} pass {k + 1}: {len(sel)} samples in {len(groups)} chunk groups")

                def one(g):
                    r0, r1, c0, c1 = row[g].min(), row[g].max() + 1, col[g].min(), col[g].max() + 1
                    w = retry(lambda: arr[t, :, r0:r1, c0:c1])
                    return g, w[:, row[g] - r0, col[g] - c0].T

                with ThreadPoolExecutor(8) as ex:
                    for n, (g, w) in enumerate(ex.map(one, groups)):
                        v[g] = w
                        if n % 200 == 0:
                            log.info(f"OlmoEarth {zn}: {n + 1}/{len(groups)} groups")
                np.save(f, v)
            got = ~(v == -128).all(axis=1)
            q[sel[got]] = v[got]
            zone_used[sel[got]] = zn
        log.info(f"OlmoEarth pass {k + 1}: still empty {(q == -128).all(axis=1).sum()}")
    df = pd.DataFrame(q, columns=[f"o{k:03d}" for k in range(128)])
    df.insert(0, "zone", zone_used)
    df.insert(0, "sid", smp.sid.to_numpy())
    df.to_parquet(PREP / "olmoearth_2019.parquet", index=False)
    log.info(f"OlmoEarth: {len(df)} samples, nodata rows {(q == -128).all(axis=1).sum()}")


if __name__ == "__main__":
    {"check": orientation_check, "aef": extract_aef, "tessera": extract_tessera,
     "olmoearth": extract_olmoearth}[sys.argv[1]]()
