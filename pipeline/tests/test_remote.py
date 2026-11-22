"""Independent re-read of stored values from the source files (network, slow).

Run with: pytest -m remote
"""
import json

import numpy as np
import pandas as pd
import pytest
import rasterio

from common import GDAL_ENV, PREP, signed, to_crs

pytestmark = pytest.mark.remote


def test_s2_values_match_rasterio_sample(samples):
    cat = pd.read_parquet(PREP / "s2/catalog.parquet").set_index("item")
    rng = np.random.default_rng(7)
    files = sorted((PREP / "s2/items").glob("*.parquet"))
    for f in rng.choice(files, 3, replace=False):
        d = pd.read_parquet(f)
        d = d[d.B04 > 0]
        if not len(d):
            continue
        d = d.sample(min(40, len(d)), random_state=1)
        r = cat.loc[f.stem]
        pts = samples.set_index("sid").loc[d.sid]
        x, y = to_crs(pts.lon, pts.lat, "EPSG:4326", f"EPSG:{r.epsg}")
        with rasterio.Env(**GDAL_ENV):
            for b in ("B04", "B12", "SCL"):
                with rasterio.open(signed(json.loads(r.hrefs)[b])) as s:
                    v = np.array([q[0] for q in s.sample(zip(x, y))])
                stored = d.scl if b == "SCL" else d[b]
                assert (v == stored.to_numpy()).all(), (f.stem, b)


def test_s1_values_match_3x3_mean(samples):
    cat = pd.read_parquet(PREP / "s1/catalog.parquet").set_index("item")
    f = sorted(p for p in (PREP / "s1/items").glob("*.parquet") if len(pd.read_parquet(p)) > 100)[0]
    d = pd.read_parquet(f).sample(30, random_state=2)
    r = cat.loc[f.stem]
    pts = samples.set_index("sid").loc[d.sid]
    x, y = to_crs(pts.lon, pts.lat, "EPSG:4326", f"EPSG:{r.epsg}")
    with rasterio.Env(**GDAL_ENV), rasterio.open(signed(json.loads(r.hrefs)["vh"])) as s:
        for xi, yi, want in zip(x, y, d.vh):
            row, col = s.index(xi, yi)
            w = s.read(1, window=((row - 1, row + 2), (col - 1, col + 2))).astype(float)
            w[(w <= 0) | (w == s.nodata)] = np.nan
            assert np.isclose(np.nanmean(w), want, rtol=1e-5)
