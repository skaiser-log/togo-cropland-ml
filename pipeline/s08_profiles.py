"""Season profiles of the training pixels: a physical check of the prepared time series.

Median NDVI (clear Sentinel-2) and VH backscatter (Sentinel-1) per month for crop and
non-crop pixels, in Savanes (one rainy season) and in the south (Plateaux, Maritime; two
seasons). Each polygon counts once: pixel values are first reduced to a polygon median.
Also counts clear Sentinel-2 observations per month, the cloud gap of the proposal.

Output: prep/checks/season_profiles.png, prep/checks/season_profiles.csv
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import FEAT, PREP

smp = pd.read_parquet(PREP / "samples.parquet")
tr = smp[smp.kind == "train"][["sid", "poly_id", "label", "region"]]
tr["area"] = tr.region.map(lambda r: "Savanes" if r == "Savanes" else ("South" if r in ("Plateaux", "Maritime") else None))
tr = tr.dropna(subset=["area"])

s2 = pd.read_parquet(FEAT / "s2_clear.parquet", columns=["sid", "date", "ndvi"]).merge(tr, on="sid")
s1 = pd.read_parquet(FEAT / "s1.parquet", columns=["sid", "date", "vh_db"]).merge(tr, on="sid")


def monthly(df, col):
    df = df.assign(month=df.date.dt.month)
    poly = df.groupby(["area", "label", "poly_id", "month"])[col].median().reset_index()
    return poly.groupby(["area", "label", "month"]).agg(median=(col, "median"), q25=(col, lambda v: v.quantile(.25)),
                                                       q75=(col, lambda v: v.quantile(.75)), n_poly=(col, "size")).reset_index()


nd, vh = monthly(s2, "ndvi"), monthly(s1, "vh_db")
clear = s2.assign(month=s2.date.dt.month).groupby(["area", "month"]).size() / tr.groupby("area").size()
pd.concat([nd.assign(var="ndvi"), vh.assign(var="vh_db")]).to_csv(PREP / "checks/season_profiles.csv", index=False)

fig, ax = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
colours = {1: "#c0841a", 0: "#4a7a5c"}
for j, area in enumerate(("Savanes", "South")):
    for i, (d, lab) in enumerate(((nd, "NDVI"), (vh, "VH (dB)"))):
        a = ax[i, j]
        for y, name in ((1, "crop"), (0, "non-crop")):
            s = d[(d.area == area) & (d.label == y)]
            a.plot(s.month, s["median"], color=colours[y], label=name, lw=2)
            a.fill_between(s.month, s.q25, s.q75, color=colours[y], alpha=.2, lw=0)
        a.set_ylabel(lab)
        a.grid(alpha=.3)
    c = clear.loc[area]
    ax[2, j].bar(c.index, c.values, color="#6b7a8f")
    ax[2, j].set_ylabel("clear S2 obs per pixel")
    ax[2, j].set_xlabel("month 2019")
    ax[0, j].set_title(area)
ax[0, 0].legend(frameon=False)
ax[2, 0].set_xticks(range(1, 13))
fig.tight_layout()
fig.savefig(PREP / "checks/season_profiles.png", dpi=110)
print(nd.pivot_table(index=["area", "label"], columns="month", values="median").round(2).to_string())
print(vh.pivot_table(index=["area", "label"], columns="month", values="median").round(1).to_string())
print(clear.unstack().round(2).to_string())
