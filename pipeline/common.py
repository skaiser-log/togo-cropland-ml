"""Shared paths, constants and raster helpers for the Togo cropland project.

Every pipeline script imports this module. Data live in ``data/`` at the repository root
(override with the environment variable ``TOGO_DATA``).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

DATA = Path(os.environ.get("TOGO_DATA", Path(__file__).resolve().parents[1] / "data")).expanduser()
RAW = DATA / "raw"
PREP = DATA / "prep"
FEAT = DATA / "features"
for _d in (PREP, FEAT):
    _d.mkdir(parents=True, exist_ok=True)

YEAR = 2019
GRID_CRS = "EPSG:32631"  # common 10 m grid; pixel centres sit at multiples of 10 m + 5
GRID_RES = 10.0
TOGO_BBOX = (-0.20, 6.05, 1.85, 11.20)  # lon/lat, a little wider than the labels

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_BANDS_10 = ["B02", "B03", "B04", "B08"]
S2_BANDS_20 = ["B05", "B06", "B07", "B8A", "B11", "B12"]
S2_BANDS = S2_BANDS_10 + S2_BANDS_20
# Scene classification codes worth reading bands for. 4 vegetation, 5 bare, 6 water,
# 2 dark area (incl. burn scars, terrain shadow), 7 unclassified. The feature step decides
# which of them count as clear; extraction keeps them all so that choice stays open.
SCL_READ = (2, 4, 5, 6, 7)
SCL_CLEAR = (4, 5, 6)

# Latitude bands used on the proposal page (section 4)
ZONES = [(6.0, 7.5, "south"), (7.5, 9.0, "centre"), (9.0, 10.3, "kara"), (10.3, 11.3, "north")]

# GDAL settings for many small range reads on cloud-optimised GeoTIFFs
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff,.TIF",
    GDAL_HTTP_MAX_RETRY="6",
    GDAL_HTTP_RETRY_DELAY="3",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_TIMEOUT="120",
    VSI_CACHE="FALSE",
)


def log_to(name: str) -> logging.Logger:
    """Logger that writes to stdout and to ``prep/logs/<name>.log``."""
    (PREP / "logs").mkdir(exist_ok=True)
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
        for h in (logging.StreamHandler(), logging.FileHandler(PREP / "logs" / f"{name}.log")):
            h.setFormatter(fmt)
            log.addHandler(h)
    return log


def zone_of(lat: np.ndarray) -> np.ndarray:
    out = np.full(len(lat), "", dtype=object)
    for lo, hi, name in ZONES:
        out[(lat >= lo) & (lat < hi)] = name
    return out


def stac_client():
    import planetary_computer as pc
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def signed(href: str) -> str:
    """Sign a Planetary Computer asset URL just before reading it.

    Tokens expire after about an hour, so long runs must not reuse the signature
    from the catalogue search. ``planetary_computer`` caches and renews tokens.
    """
    import planetary_computer as pc

    return pc.sign(href)


def retry(fn, tries: int = 4, wait: float = 10.0):
    for k in range(tries):
        try:
            return fn()
        except Exception:  # network errors surface as many different types
            if k == tries - 1:
                raise
            time.sleep(wait * (k + 1))


def read_points(src: rasterio.io.DatasetReader, rows: np.ndarray, cols: np.ndarray,
                bands: list[int] | None = None, block: int | None = None,
                halo: int = 0) -> np.ndarray:
    """Read pixel values at (row, col) positions, one window per internal block.

    Points are grouped by the internal tile they fall in and each group is read
    as the smallest window covering its points (plus ``halo`` pixels for a moving
    window). That keeps the number of HTTP range requests near the number of
    blocks touched, instead of one per point.

    Returns an array of shape (n_points, n_bands, 2*halo+1, 2*halo+1); positions
    outside the raster stay NaN.
    """
    bands = bands or list(range(1, src.count + 1))
    block = block or src.block_shapes[0][0]
    k = 2 * halo + 1
    out = np.full((len(rows), len(bands), k, k), np.nan, dtype="float32")
    inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
    idx = np.flatnonzero(inside)
    if not len(idx):
        return out
    key = (rows[idx] // block) * 100_000 + cols[idx] // block
    order = np.argsort(key, kind="stable")
    idx, key = idx[order], key[order]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(idx)]):
        g = idx[a:b]
        r0, r1 = rows[g].min() - halo, rows[g].max() + halo + 1
        c0, c1 = cols[g].min() - halo, cols[g].max() + halo + 1
        # clip the window to the raster and pad by hand; boundless reads are slow remotely
        cr0, cr1 = max(r0, 0), min(r1, src.height)
        cc0, cc1 = max(c0, 0), min(c1, src.width)
        part = src.read(bands, window=Window(cc0, cr0, cc1 - cc0, cr1 - cr0), out_dtype="float32")
        if src.nodata is not None and not np.isnan(src.nodata):
            part[part == src.nodata] = np.nan
        arr = np.full((len(bands), r1 - r0, c1 - c0), np.nan, dtype="float32")
        arr[:, cr0 - r0:cr1 - r0, cc0 - c0:cc1 - c0] = part
        rr, cc = rows[g] - r0, cols[g] - c0
        for dr in range(k):
            for dc in range(k):
                out[g, :, dr, dc] = arr[:, rr + dr - halo, cc + dc - halo].T
    return out


def rowcol(src, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row/col of coordinates in ``src``'s CRS; works for north-up and bottom-up grids."""
    inv = ~src.transform
    c, r = inv @ (np.asarray(xs, float), np.asarray(ys, float))
    return np.floor(r).astype(int), np.floor(c).astype(int)


def to_crs(xs, ys, src_crs, dst_crs):
    from pyproj import Transformer

    t = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return t.transform(np.asarray(xs), np.asarray(ys))
