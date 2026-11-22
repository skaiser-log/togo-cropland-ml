"""Fields of the World (FTW) as a third benchmark, over Savanes on Sentinel-2 tile 30PZS.

FTW runs in its own environment (for example ``.venv-ftw``, ftw-tools 1.4.3 with
torchgeo pinned below 0.8, because ftw 1.4.3 imports ``AugmentationSequential``, which
torchgeo 0.10 removed). Commands, run from ``prep/ftw/run``:

    ftw inference download \
        --win_a S2A_MSIL2A_20190417T102031_R065_T30PZS_20201007T030843 \
        --win_b S2A_MSIL2A_20191123T102341_R065_T30PZS_20201003T191119 \
        --bbox="-0.0902,9.884,0.7465,10.8411" -o savanes_30PZS_stack.tif
    ftw inference run savanes_30PZS_stack.tif -m ../3_Class_FULL_FTW_Pretrained.ckpt \
        -o savanes_30PZS_ftw.tif

Scene choice: relative orbit 65, because orbit 22 covers only the eastern 7 % of the tile;
17 April (1.4 % cloud, land preparation) and 23 November (0.06 %, harvest) are the clearest
dates near the start and end of the Savanes season. The bounding box is Savanes within the
tile. The 2019 scenes are processing baseline 02.12, which needs no offset; ftw warns that
it is untested.

FTW predicts 0 background, 1 field interior, 2 field boundary. Field and boundary both
count as cropland here. Scores use the test points inside the window and are written next
to the other benchmarks for the same points.

Output: prep/checks/ftw_scores.csv, prep/checks/ftw_classes_at_labels.csv
"""
import numpy as np
import pandas as pd
import rasterio

from common import PREP, rowcol, to_crs
from s05_benchmarks import scores

RUN = PREP / "ftw/run"
CHECKS = PREP / "checks"
EDGE = 64  # pixels discarded at the window edge, where ftw pads the input


def sample_ftw(smp: pd.DataFrame) -> pd.Series:
    with rasterio.open(RUN / "savanes_30PZS_ftw.tif") as s:
        x, y = to_crs(smp.lon, smp.lat, "EPSG:4326", s.crs)
        r, c = rowcol(s, x, y)
        ok = (r >= EDGE) & (r < s.height - EDGE) & (c >= EDGE) & (c < s.width - EDGE)
        band = s.read(1)
        with rasterio.open(RUN / "savanes_30PZS_stack.tif") as st:
            valid = st.read(1) > 0  # outside the scene footprint the input is 0
    v = np.full(len(smp), np.nan)
    v[ok] = np.where(valid[r[ok], c[ok]], band[r[ok], c[ok]], np.nan)
    return pd.Series(v, index=smp.index)


def main():
    smp = pd.read_parquet(PREP / "samples.parquet")
    smp["ftw"] = sample_ftw(smp)
    inside = smp[smp.ftw.notna()]
    # what the classes mean on our labels
    ct = pd.crosstab([inside.kind, inside.label], inside.ftw.astype(int), normalize="index")
    ct.to_csv(CHECKS / "ftw_classes_at_labels.csv")
    print(ct.round(3))

    bench = pd.read_parquet(PREP / "benchmarks.parquet")
    t = inside[inside.kind == "test"].merge(bench, on="sid")
    t["dea_mask"] = t.dea_mask.fillna(0)
    preds = {"Fields of the World": t.ftw >= 1, "WorldCover 2020": t.wc2020 == 40,
             "WorldCover 2021": t.wc2021 == 40, "DEA crop mask 2019": t.dea_mask == 1}
    rows = []
    for name, p in preds.items():
        for subset, sel in (("all", t.label.notna()), ("unanimous", t.agreement == "unanimous"),
                            ("split", t.agreement == "split")):
            if sel.sum():
                rows.append(dict(map=name, subset=subset, window="Savanes 30PZS",
                                 **scores(t.label[sel].to_numpy(), p[sel].astype(int).to_numpy())))
    df = pd.DataFrame(rows)
    df.to_csv(CHECKS / "ftw_scores.csv", index=False)
    print(df.round(3).to_string(index=False))
    with rasterio.open(RUN / "savanes_30PZS_ftw.tif") as s:
        a = s.read(1)[EDGE:-EDGE, EDGE:-EDGE]
    print("class shares in the window:", np.round(np.bincount(a.ravel(), minlength=3) / a.size, 3))


if __name__ == "__main__":
    main()
