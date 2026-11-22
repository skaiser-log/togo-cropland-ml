"""Reference labels: training polygons, test points and the sample table.

Inputs (NASA Harvest / UMD, Kerner et al. 2020, Zenodo 3836629 and 4680394):
  raw/crop_merged_v2, raw/noncrop_merged_v2   training polygons
  raw/togo_test_majority                      306 random test points, majority label
  raw/togo_points/togo/...                    the three labellers' votes on the random sample

Outputs:
  prep/labels.gpkg   layers train_polygons, test_points
  prep/samples.parquet  one row per location features are extracted for:
      every 10 m grid pixel whose centre lies inside a training polygon (kind=train),
      plus each test point at its exact position (kind=test).
"""
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from common import GRID_CRS, GRID_RES, PREP, RAW, zone_of

R = RAW / "togo_points/togo/"

# --- training polygons -------------------------------------------------------------------
crop = gpd.read_file(RAW / "crop_merged_v2/crop_merged_v2/crop_merged_v2.shp").assign(label=1)
non = gpd.read_file(RAW / "noncrop_merged_v2/noncrop_merged_v2/noncrop_merged_v2.shp").assign(label=0)
poly = pd.concat([crop, non], ignore_index=True)[["label", "geometry"]]
n_invalid = int((~poly.is_valid).sum())
poly["geometry"] = poly.geometry.make_valid()
poly = gpd.GeoDataFrame(poly, crs=crop.crs).to_crs(GRID_CRS)
poly.insert(0, "poly_id", np.arange(len(poly), dtype="int32"))
poly["area_ha"] = poly.area / 1e4
cen = poly.geometry.centroid.to_crs(4326)
poly["zone"] = zone_of(cen.y.to_numpy())

reg = gpd.read_file(PREP / "regions.gpkg", layer="regions").to_crs(GRID_CRS)
# 117 polygons lie up to 5 km across the border (Ghana, Benin); they keep the nearest region
rp = gpd.GeoDataFrame(geometry=poly.geometry.representative_point(), crs=GRID_CRS)
poly["in_togo"] = rp.within(reg.union_all()).to_numpy()
poly["region"] = gpd.sjoin_nearest(rp, reg[["region", "geometry"]], how="left")["region"].groupby(level=0).first().to_numpy()

# --- test points with the three labellers' votes ------------------------------------------
test = gpd.read_file(RAW / "togo_test_majority/togo_test_majority/togo_test_majority.shp")
test["label"] = test.label.astype(int)
bb = gpd.read_file(R / "BB_random_sample_1k/random_sample_1k_BB.shp")
bm = gpd.read_file(R / "random_sample_bm/BM_Labels.shp")
cn = gpd.read_file(R / "random_sample_cn/random_sample_1k_cn.shp")
three = bb[["id", "bb_label", "geometry"]].merge(bm[["id", "bm_labels"]], on="id").merge(cn[["id", "cn_labels"]], on="id")
three["votes"] = three[["bb_label", "bm_labels", "cn_labels"]].astype(int).sum(axis=1)
three = gpd.GeoDataFrame(three, crs=bb.crs).to_crs(GRID_CRS)
test = test.to_crs(GRID_CRS)
test = gpd.sjoin_nearest(test, three[["id", "votes", "geometry"]], max_distance=1.0, how="left",
                         distance_col="match_m").drop(columns="index_right")
test = test[~test.index.duplicated()].reset_index(drop=True)
test.insert(0, "test_id", np.arange(len(test), dtype="int32"))
test["votes"] = test.votes.astype("Int64")
test["agreement"] = np.where(test.votes.isin([0, 3]), "unanimous", "split")
test["agreement"] = test.agreement.where(test.votes.notna(), None)
ll = test.to_crs(4326)
test["zone"] = zone_of(ll.geometry.y.to_numpy())
test["region"] = gpd.sjoin(test[["geometry"]], reg[["region", "geometry"]], how="left",
                           predicate="within")["region"].to_numpy()
# distance from each test point to the nearest training polygon (independence)
test["dist_train_m"] = test.geometry.apply(lambda p: poly.distance(p).min())

# --- training pixels on the common 10 m grid ----------------------------------------------
rows = []
for p in poly.itertuples():
    g = p.geometry
    x0, y0, x1, y1 = g.bounds
    xs = np.arange(np.floor(x0 / GRID_RES), np.ceil(x1 / GRID_RES)) * GRID_RES + GRID_RES / 2
    ys = np.arange(np.floor(y0 / GRID_RES), np.ceil(y1 / GRID_RES)) * GRID_RES + GRID_RES / 2
    X, Y = np.meshgrid(xs, ys)
    cand = gpd.GeoSeries.from_xy(X.ravel(), Y.ravel(), crs=GRID_CRS)
    inside = cand.within(g).to_numpy()
    fallback = not inside.any()
    if fallback:  # polygon smaller than a pixel centre spacing: take the pixel under its centre
        rp = g.representative_point()
        X = np.array([[np.floor(rp.x / GRID_RES) * GRID_RES + GRID_RES / 2]])
        Y = np.array([[np.floor(rp.y / GRID_RES) * GRID_RES + GRID_RES / 2]])
        inside = np.array([True])
    cx, cy = X.ravel()[inside], Y.ravel()[inside]
    h = GRID_RES / 2
    cover = np.array([box(x - h, y - h, x + h, y + h).intersection(g).area / GRID_RES**2 for x, y in zip(cx, cy)])
    rows.append(pd.DataFrame(dict(poly_id=p.poly_id, label=p.label, e=cx, n=cy, cover=cover,
                                  centre_fallback=fallback)))
pix = pd.concat(rows, ignore_index=True)

# a pixel claimed by two polygons: keep it once if the labels agree, drop it if they conflict
dup = pix.duplicated(["e", "n"], keep=False)
conflict = pix[dup].groupby(["e", "n"]).label.nunique()
conflict_keys = set(conflict[conflict > 1].index)
n_conflict = len(conflict_keys)
pix = pix[~pix.set_index(["e", "n"]).index.isin(conflict_keys)]
pix = pix.sort_values("cover", ascending=False).drop_duplicates(["e", "n"]).sort_values(["poly_id", "n", "e"])
pix = pix.merge(poly[["poly_id", "zone", "region", "in_togo"]], on="poly_id")
pix["kind"] = "train"

tp = test.assign(e=test.geometry.x, n=test.geometry.y, kind="test", in_togo=True, cover=np.nan, centre_fallback=False)
cols = ["kind", "poly_id", "test_id", "label", "votes", "agreement", "e", "n", "cover",
        "centre_fallback", "zone", "region", "in_togo"]
samples = pd.concat([pix, tp.drop(columns="geometry")], ignore_index=True).reindex(columns=cols)
samples["poly_id"] = samples.poly_id.astype("Int32")
samples["test_id"] = samples.test_id.astype("Int32")
lon, lat = gpd.GeoSeries.from_xy(samples.e, samples.n, crs=GRID_CRS).to_crs(4326).pipe(lambda s: (s.x, s.y))
samples["lon"], samples["lat"] = lon.to_numpy(), lat.to_numpy()
# 5 km cells: a stricter grouping for cross-validation than the polygon (spatial clusters)
# A training polygon takes the cell of its pixel centroid as a whole: 27 large polygons cross a
# cell border, and a per-pixel cell would put parts of one field on both sides of a fold.
cx = samples.e.where(samples.kind == "test", samples.groupby("poly_id").e.transform("mean"))
cy = samples.n.where(samples.kind == "test", samples.groupby("poly_id").n.transform("mean"))
samples["cell5km"] = (cx // 5000).astype(int).astype(str) + "_" + (cy // 5000).astype(int).astype(str)
samples.insert(0, "sid", np.arange(len(samples), dtype="int32"))
samples.to_parquet(PREP / "samples.parquet", index=False)

poly.to_file(PREP / "labels.gpkg", layer="train_polygons")
test.drop(columns=["id"]).to_file(PREP / "labels.gpkg", layer="test_points")

# --- report ----------------------------------------------------------------------------------
tr = samples[samples.kind == "train"]
print(f"polygons: {len(poly)} ({(poly.label == 1).sum()} crop, {(poly.label == 0).sum()} non-crop), "
      f"{n_invalid} invalid geometries repaired")
print(f"area ha crop {poly[poly.label == 1].area_ha.sum():.0f}, non-crop {poly[poly.label == 0].area_ha.sum():.0f}; "
      f"crop median {poly[poly.label == 1].area_ha.median():.3f} ha")
print(f"training pixels: {len(tr)} ({(tr.label == 1).sum()} crop, {(tr.label == 0).sum()} non-crop); "
      f"pure (cover=1) {(tr.cover > 0.999).mean():.0%}; polygons with centre fallback "
      f"{tr[tr.centre_fallback].poly_id.nunique()}; conflicting pixels dropped {n_conflict}")
print("pixels per polygon: median", int(tr.groupby("poly_id").size().median()),
      "max", int(tr.groupby("poly_id").size().max()))
print(f"test points: {len(test)} ({(test.label == 1).sum()} crop); votes matched {test.votes.notna().sum()}; "
      f"unanimous {(test.agreement == 'unanimous').sum()}, split {(test.agreement == 'split').sum()}; "
      f"within 500 m of a training polygon {(test.dist_train_m < 500).sum()}")
print("majority label consistent with votes:", bool(((test.votes >= 2).astype(int) == test.label).all()))
print("\nby region (polygons crop/non, test points, test crop share):")
for r in reg.region:
    pr, tr_ = poly[poly.region == r], test[test.region == r]
    print(f"  {r:9s} {(pr.label == 1).sum():4d} / {(pr.label == 0).sum():4d}   {len(tr_):3d}   {tr_.label.mean():.0%}")
print("polygons outside Togo (nearest region kept):", (~poly.in_togo).sum(), "- test points without region:", test.region.isna().sum())
