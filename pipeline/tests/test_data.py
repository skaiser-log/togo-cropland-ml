"""Checks on the prepared data, run before the modelling pipeline.

Numbers asserted here are the ones on the proposal page and in the 29 October data check;
a failure means the preparation drifted from what the proposal claims.
"""
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from common import FEAT, PREP, S2_BANDS

# ---------------------------------------------------------------- labels


def test_label_counts_match_proposal(samples):
    poly = gpd.read_file(PREP / "labels.gpkg", layer="train_polygons")
    assert (poly.label == 1).sum() == 688 and (poly.label == 0).sum() == 588
    t = samples[samples.kind == "test"]
    assert len(t) == 306 and (t.label == 1).sum() == 106
    assert (t.agreement == "unanimous").sum() == 237 and (t.agreement == "split").sum() == 69


def test_majority_label_matches_votes(samples):
    t = samples[samples.kind == "test"]
    assert ((t.votes >= 2).astype(int) == t.label).all()


def test_test_points_independent_of_training():
    t = gpd.read_file(PREP / "labels.gpkg", layer="test_points")
    assert (t.dist_train_m < 500).sum() == 6
    assert (t.dist_train_m > 0).all(), "a test point lies inside a training polygon"


def test_training_pixels_on_common_grid(samples):
    tr = samples[samples.kind == "train"]
    assert np.allclose((tr.e - 5) % 10, 0) and np.allclose((tr.n - 5) % 10, 0)
    assert not tr.duplicated(["e", "n"]).any()
    poly = gpd.read_file(PREP / "labels.gpkg", layer="train_polygons")
    assert set(tr.poly_id) == set(poly.poly_id), "a polygon has no training pixel"
    assert tr.cover.between(0, 1.0001).all()


def test_sample_ids_unique_and_kinds_disjoint(samples):
    assert samples.sid.is_unique
    assert samples[samples.kind == "train"].test_id.isna().all()
    assert samples[samples.kind == "test"].poly_id.isna().all()


# ---------------------------------------------------------------- Sentinel-2


def test_all_s2_items_extracted():
    cat = pd.read_parquet(PREP / "s2/catalog.parquet")
    home = pd.read_parquet(PREP / "s2/home_tile.parquet")
    need = set(cat[cat.tile.isin(set(home.tile))].item)
    done = {f.stem for f in (PREP / "s2/items").glob("*.parquet")}
    assert need <= done, f"{len(need - done)} items missing, rerun s02_s2_extract.py"


def test_every_sample_has_clear_s2_observations(samples):
    cnt = pd.read_parquet(FEAT / "s2_counts.parquet").reindex(samples.sid)
    assert cnt.n_seen.notna().all(), "a sample was never covered by Sentinel-2"
    assert (cnt.n_clear.fillna(0) >= 5).mean() > 0.99
    assert cnt.loc[samples[samples.kind == "test"].sid, "n_clear"].min() >= 5


def test_s2_reflectance_plausible():
    c = pd.read_parquet(FEAT / "s2_clear.parquet", columns=S2_BANDS + ["ndvi", "date", "scl"])
    assert c.date.dt.year.eq(2019).all()
    med = c[c.scl == 4][S2_BANDS].median()
    assert 0.02 < med.B02 < 0.12 and med.B08 > med.B04, med.to_dict()  # vegetation: NIR above red
    assert c[c.scl == 4].ndvi.median() > 0.4
    assert c[c.scl == 6].ndvi.median() < 0.1  # water


def test_rainy_season_is_cloudy_for_optical():
    # Pixel-wise masking still gives most samples one clear look in July, but the depth of
    # coverage collapses: about one clear observation per month against five in the dry season.
    c = pd.read_parquet(FEAT / "s2_clear.parquet", columns=["sid", "date"])
    per_sample = c.groupby(c.date.dt.month).size() / c.sid.nunique()
    assert per_sample.loc[[7, 8]].mean() < 0.4 * per_sample.loc[[1, 12]].mean()


# ---------------------------------------------------------------- Sentinel-1


def test_all_s1_scenes_extracted():
    cat = pd.read_parquet(PREP / "s1/catalog.parquet")
    done = {f.stem for f in (PREP / "s1/items").glob("*.parquet")}
    assert set(cat.item) <= done


def test_s1_coverage_and_values(samples):
    s1 = pd.read_parquet(FEAT / "s1.parquet")
    n = s1.groupby("sid").size().reindex(samples.sid).fillna(0)
    assert (n >= 20).mean() > 0.99, n.describe()
    assert -14 < s1.vv_db.median() < -4 and -22 < s1.vh_db.median() < -10
    assert (s1.vh_db < s1.vv_db).mean() > 0.9
    months = s1.groupby(s1.date.dt.month).sid.nunique()
    assert len(months) == 12 and months.min() > 0.8 * len(samples)


# ---------------------------------------------------------------- embeddings


def test_aef_orientation_check_passed():
    chk = json.loads((PREP / "checks/aef_orientation.json").read_text())
    assert chk["passed"], chk


def test_aef_vectors_unit_length(feat, samples):
    d = feat("D")
    norm = np.sqrt((d ** 2).sum(axis=1, min_count=64))
    valid = norm.dropna()
    assert len(valid) / len(d) > 0.99
    assert valid.between(0.97, 1.03).mean() > 0.99
    assert d.loc[samples[samples.kind == "test"].sid].notna().all(axis=1).all()


def test_tessera_complete(feat, samples):
    e = feat("E")
    assert e.shape[1] == 128
    assert e.notna().all(axis=1).mean() > 0.99
    assert e.loc[samples[samples.kind == "test"].sid].notna().all(axis=1).all()


# ---------------------------------------------------------------- benchmarks


def test_worldcover_reproduces_data_check():
    s = pd.read_csv(PREP / "checks/benchmark_scores.csv")
    r = s[(s["map"] == "WorldCover 2020") & (s.subset == "all")].iloc[0]
    assert (round(r.OA, 2), round(r.UA_crop, 2), round(r.PA_crop, 2)) == (0.81, 0.88, 0.53)
    r = s[(s["map"] == "WorldCover 2021") & (s.subset == "all")].iloc[0]
    assert round(r.PA_crop, 2) == 0.58 or round(r.PA_crop, 2) == 0.59


def test_benchmarks_cover_test_points(samples):
    b = pd.read_parquet(PREP / "benchmarks.parquet").set_index("sid").loc[samples[samples.kind == "test"].sid]
    assert b.wc2020.notna().all() and b.wc2021.notna().all()
    assert b.dea_mask.isna().sum() <= 5  # DEA masks five non-crop points itself (value 255)


# ---------------------------------------------------------------- features


@pytest.mark.parametrize("s", list("ABCDEF"))
def test_feature_set_aligned_and_clean(feat, samples, s):
    df = feat(s)
    cols = json.loads((FEAT / "columns.json").read_text())[s]
    assert list(df.index) == list(samples.sid), "feature rows not aligned with samples"
    X = df[cols]
    forbidden = {"label", "votes", "agreement", "poly_id", "test_id", "e", "n", "lon", "lat", "kind"}
    assert not any(c.split("_", 1)[-1] in forbidden for c in cols)
    assert np.isfinite(X.to_numpy(dtype="float64")[~np.isnan(X.to_numpy(dtype="float64"))]).all()
    # set A has no clear view between October and December for 37 polygons (1.1 % of pixels)
    assert X.isna().all(axis=1).mean() < (0.02 if s == "A" else 0.01), "too many samples without any feature"
    assert X.loc[samples[samples.kind == "test"].sid].notna().any(axis=1).all(), "a test point has no features"
    assert not X.isna().all(axis=0).any(), "a feature column is entirely empty"


def test_set_c_contains_set_b():
    cols = json.loads((FEAT / "columns.json").read_text())
    assert set(cols["B"]) < set(cols["C"])


def test_set_a_single_date_mostly(feat):
    a = feat("A")
    assert (a.a_gap_days == 0).mean() > 0.6
    assert a.a_gap_days.max() <= 92
