"""Feature sets A to E from the extracted time series and embeddings.

  A  one clear image: per home MGRS tile the Sentinel-2 scene with the most clear samples
     between 1 October and 31 December 2019 (after the rains, the first clear window);
     samples cloudy in that scene take their clear observation nearest in time inside
     the same window. 10 bands + 4 indices.
  B  optical season: every clear Sentinel-2 observation of 2019. Annual percentiles of
     bands and indices, seasonal medians of NDVI and NDMI, NDVI harmonic fit.
  C  B plus Sentinel-1: monthly medians of VV, VH and VH/VV (dB) and annual statistics.
  D  AlphaEarth 2019 embedding, 64 dimensions (dequantised).
  E  TESSERA 2019 embedding, 128 dimensions.

Clear means SCL 4 (vegetation), 5 (bare soil) or 6 (water). Missing values (no clear
observation in a season, for example) stay NaN; scikit-learn's trees handle NaN.
Coordinates, observation counts and label columns are never features.

Outputs: features/set_A.parquet ... set_F.parquet (sid + features; F = OlmoEarth v1.3, added 2 Nov),
         features/s2_clear.parquet, features/s1.parquet (long tables for plots),
         features/columns.json
"""
import json

import numpy as np
import pandas as pd

from common import FEAT, PREP, S2_BANDS, SCL_CLEAR, log_to
from s04_embeddings import dequantise_aef

log = log_to("features")
A_WINDOW = ("2019-10-01", "2019-12-31")
SEASONS = {"djf": (1, 3), "amj": (4, 6), "jas": (7, 9), "ond": (10, 12)}  # Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec
INDICES = {"ndvi": ("B08", "B04"), "ndmi": ("B08", "B11"), "ndwi": ("B03", "B08"), "ndre": ("B8A", "B05")}


def s2_long() -> pd.DataFrame:
    cat = pd.read_parquet(PREP / "s2/catalog.parquet")[["item", "tile", "datetime"]]
    parts = []
    for f in sorted((PREP / "s2/items").glob("*.parquet")):
        d = pd.read_parquet(f)
        if len(d):
            parts.append(d.assign(item=f.stem))
    s2 = pd.concat(parts, ignore_index=True).merge(cat, on="item")
    s2["date"] = s2.datetime.dt.normalize()
    clear = s2[s2.scl.isin(SCL_CLEAR) & (s2[S2_BANDS] > 0).all(axis=1) & (s2[S2_BANDS] < 10000).all(axis=1)].copy()
    # one acquisition can appear twice (reprocessed duplicates); keep one row per sample and day
    clear = clear.drop_duplicates(["sid", "date"])
    clear[S2_BANDS] = clear[S2_BANDS].astype("float32") / 10000.0  # baseline 02.12: no BOA offset
    for k, (a, b) in INDICES.items():
        clear[k] = ((clear[a] - clear[b]) / (clear[a] + clear[b])).astype("float32")
    log.info(f"S2: {len(s2)} observations, {len(clear)} clear on {clear.sid.nunique()} samples")
    seen = s2.drop_duplicates(["sid", "date"]).groupby("sid").size().rename("n_seen")
    return clear, seen


def set_a(clear: pd.DataFrame) -> pd.DataFrame:
    win = clear[(clear.date >= A_WINDOW[0]) & (clear.date <= A_WINDOW[1])]
    best = win.groupby(["tile", "item"]).size().reset_index(name="n").sort_values("n").groupby("tile").tail(1)
    best = best.merge(win[["item", "date"]].drop_duplicates("item"), on="item")
    w = win.merge(best[["tile", "date"]].rename(columns={"date": "best_date"}), on="tile")
    w["gap_days"] = (w.date - w.best_date).dt.days.abs()
    w = w.sort_values(["sid", "gap_days"]).drop_duplicates("sid")
    cols = S2_BANDS + list(INDICES)
    out = w.set_index("sid")[cols].add_prefix("a_")
    out["a_gap_days"] = w.set_index("sid").gap_days  # diagnostic, dropped before modelling
    log.info(f"set A: scenes {dict(zip(best.tile, best.date.dt.strftime('%m-%d')))}; "
             f"{(w.gap_days == 0).mean():.0%} of samples clear on the chosen date")
    return out


def harmonic(g: pd.DataFrame) -> pd.Series:
    t = 2 * np.pi * (g.date.dt.day_of_year.to_numpy() - 1) / 365.0
    if len(t) < 8:
        return pd.Series(np.nan, index=["c0", "a1", "b1", "a2", "b2"])
    X = np.column_stack([np.ones_like(t), np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)])
    coef, *_ = np.linalg.lstsq(X, g.ndvi.to_numpy(), rcond=None)
    return pd.Series(coef, index=["c0", "a1", "b1", "a2", "b2"])


def set_b(clear: pd.DataFrame) -> pd.DataFrame:
    g = clear.groupby("sid")
    parts = []
    for q in (10, 50, 90):
        parts.append(g[S2_BANDS].quantile(q / 100).add_prefix("b_").add_suffix(f"_p{q}"))
    for q in (10, 25, 50, 75, 90):
        parts.append(g[list(INDICES)].quantile(q / 100).add_prefix("b_").add_suffix(f"_p{q}"))
    parts.append(g[list(INDICES)].std().add_prefix("b_").add_suffix("_std"))
    m = clear.date.dt.month
    for s, (lo, hi) in SEASONS.items():
        sub = clear[(m >= lo) & (m <= hi)].groupby("sid")[["ndvi", "ndmi"]].median()
        parts.append(sub.add_prefix("b_").add_suffix(f"_{s}"))
    parts.append(g[["date", "ndvi"]].apply(harmonic).add_prefix("b_ndvi_h_"))
    out = pd.concat(parts, axis=1)
    out["b_ndvi_range"] = out.b_ndvi_p90 - out.b_ndvi_p10
    return out


def s1_long() -> pd.DataFrame:
    cat = pd.read_parquet(PREP / "s1/catalog.parquet")[["item", "datetime", "rel_orbit"]]
    parts = [pd.read_parquet(f).assign(item=f.stem) for f in sorted((PREP / "s1/items").glob("*.parquet"))]
    s1 = pd.concat([p for p in parts if len(p)], ignore_index=True).merge(cat, on="item")
    s1["date"] = s1.datetime.dt.normalize()
    # neighbouring frames of one pass overlap: average the duplicates of one sample and day
    s1 = s1.groupby(["sid", "date", "rel_orbit"], as_index=False)[["vv", "vh"]].mean()
    s1["vv_db"] = (10 * np.log10(s1.vv)).astype("float32")
    s1["vh_db"] = (10 * np.log10(s1.vh)).astype("float32")
    s1["ratio_db"] = s1.vh_db - s1.vv_db
    log.info(f"S1: {len(s1)} observations on {s1.sid.nunique()} samples")
    return s1


def set_c_radar(s1: pd.DataFrame) -> pd.DataFrame:
    v = ["vv_db", "vh_db", "ratio_db"]
    s1 = s1.assign(month=s1.date.dt.month)
    mon = s1.pivot_table(index="sid", columns="month", values=v, aggfunc="median")
    mon.columns = [f"c_{a}_m{b:02d}" for a, b in mon.columns]
    g = s1.groupby("sid")[v]
    stats = pd.concat([g.quantile(.1).add_suffix("_p10"), g.median().add_suffix("_p50"),
                       g.quantile(.9).add_suffix("_p90"), g.std().add_suffix("_std")], axis=1).add_prefix("c_")
    return pd.concat([mon, stats], axis=1)


def main():
    smp = pd.read_parquet(PREP / "samples.parquet")[["sid"]].set_index("sid")
    clear, seen = s2_long()
    clear.drop(columns=["item"]).to_parquet(FEAT / "s2_clear.parquet", index=False)
    a, b = set_a(clear), set_b(clear)
    s1 = s1_long()
    s1.to_parquet(FEAT / "s1.parquet", index=False)
    c = pd.concat([b, set_c_radar(s1)], axis=1)
    aef = pd.read_parquet(PREP / "aef_2019.parquet").set_index("sid")
    d = pd.DataFrame(dequantise_aef(aef[[f"a{k:02d}" for k in range(64)]].to_numpy()),
                     index=aef.index, columns=[f"d_{k:02d}" for k in range(64)])
    e = pd.read_parquet(PREP / "tessera_2019.parquet").set_index("sid").add_prefix("e_")
    olm = pd.read_parquet(PREP / "olmoearth_2019.parquet").set_index("sid")
    f = pd.DataFrame(dequantise_aef(olm[[f"o{k:03d}" for k in range(128)]].to_numpy()),  # same scheme as AlphaEarth
                     index=olm.index, columns=[f"f_{k:03d}" for k in range(128)])
    cols = {}
    for name, df in zip("ABCDEF", (a, b, c, d, e, f)):
        df = smp.join(df)  # every sample present, NaN where a source has nothing
        df.reset_index().to_parquet(FEAT / f"set_{name}.parquet", index=False)
        feats = [x for x in df.columns if not x.endswith("_gap_days")]
        cols[name] = feats
        nan = df[feats].isna().mean()
        log.info(f"set {name}: {len(feats)} features; rows without any value {df[feats].isna().all(axis=1).sum()}; "
                 f"median NaN share {nan.median():.1%}, worst {nan.idxmax()} {nan.max():.1%}")
    seen.to_frame().join(clear.groupby("sid").size().rename("n_clear")).to_parquet(FEAT / "s2_counts.parquet")
    (FEAT / "columns.json").write_text(json.dumps(cols, indent=1))


if __name__ == "__main__":
    main()
