"""Checks on the modelling design: no test point in training, spatially separate folds."""
import json

import numpy as np
import pytest
from sklearn.model_selection import GroupKFold

from common import PREP
from s09_tune import SETS, VAL_CAP, cap_rows, training_table


@pytest.fixture(scope="module")
def table():
    return training_table()


def test_training_table_has_no_test_points(table, samples):
    meta, X = table
    test_sids = set(samples.loc[samples.kind == "test", "sid"])
    assert not test_sids & set(meta.index)
    assert all(X[s].index.equals(meta.index) for s in SETS)


def test_training_rows_have_values_in_every_set(table):
    meta, X = table
    for s in SETS:
        assert not X[s].isna().all(axis=1).any(), s
    assert len(meta) > 0.98 * 71_660


def test_folds_never_share_a_cell_or_polygon(table):
    meta, _ = table
    for tr, va in GroupKFold(5).split(meta, groups=meta.cell5km):
        assert not set(meta.cell5km.iloc[tr]) & set(meta.cell5km.iloc[va])
        assert not set(meta.poly_id.iloc[tr]) & set(meta.poly_id.iloc[va])


def test_cap_rows_limits_pixels_per_polygon(table):
    meta, _ = table
    rows = cap_rows(meta, VAL_CAP, 0)
    assert meta.loc[rows].groupby("poly_id").size().max() <= VAL_CAP
    assert meta.loc[rows].poly_id.nunique() == meta.poly_id.nunique()


def test_common_setting_close_to_every_best():
    path = PREP / "checks/tune_best.json"
    if not path.exists():
        pytest.skip("s09_tune.py has not run")
    best = json.loads(path.read_text())
    assert best["common"]["worst_gap"] < 0.01
    assert np.isclose(max(best[s]["F1_crop"] for s in SETS), max(best["D"]["F1_crop"], best["E"]["F1_crop"]))


def test_test_predictions_complete():
    import pandas as pd

    path = PREP / "test_predictions.parquet"
    if not path.exists():
        pytest.skip("s12_test.py has not run")
    p = pd.read_parquet(path)
    assert len(p) == 306 and p.sid.is_unique
    maps = [c for c in p.columns if c.startswith(("RF ", "Tree", "WorldCover", "DEA")) and not c.endswith("p_crop")]
    assert len(maps) == 9 and p[maps].isin([0, 1]).all().all()


@pytest.mark.parametrize(("stem", "col", "src"), [("d_crop", "RF D", "test_predictions.parquet"),
                                                   ("f_crop", "RF F", "test_predictions_F.parquet")])
def test_forest_maps_reproduce_test_predictions(stem, col, src):
    """The wall-to-wall maps (s13, s17) read at the test points give the forest's test predictions."""
    import pandas as pd
    import rasterio

    from s14_areas import at_points

    p = pd.read_parquet(PREP / "test_predictions.parquet")
    if not (PREP / src).exists():
        pytest.skip(f"{src} missing")
    p = p.merge(pd.read_parquet(PREP / src)[["test_id", col]].rename(columns={col: "pred"}), on="test_id")
    for region in ("Savanes", "Plateaux"):
        f = PREP / "maps" / f"{stem}_{region}.tif"
        if not f.exists():
            pytest.skip(f"{f.name} missing")
        with rasterio.open(f) as s:
            a, tr = s.read(1), s.transform
        pts = p[p.region == region]
        assert (at_points(a, tr, pts.lon, pts.lat) == pts.pred.to_numpy()).all(), region
