"""Administrative regions of Togo (geoBoundaries ADM1, CC BY 4.0).

Output: prep/regions.gpkg with the five regions. Savanes and Plateaux are the two
regions mapped wall to wall.
"""
import geopandas as gpd
import requests

from common import PREP, RAW

URL = "https://www.geoboundaries.org/api/current/gbOpen/TGO/ADM1/"

meta = requests.get(URL, timeout=60).json()
src = RAW / "geoBoundaries-TGO-ADM1.geojson"
if not src.exists():
    src.write_bytes(requests.get(meta["gjDownloadURL"], timeout=120).content)
reg = gpd.read_file(src)[["shapeName", "geometry"]].rename(columns={"shapeName": "region"})
reg["region"] = reg.region.str.replace(" Region", "", regex=False).str.strip()
reg["area_km2"] = reg.to_crs(32631).area / 1e6
reg["mapped"] = reg.region.isin(["Savanes", "Plateaux"])
reg.to_file(PREP / "regions.gpkg", layer="regions")
print(reg.drop(columns="geometry").round(0).to_string(index=False))
print("source:", meta["gjDownloadURL"], "year", meta.get("boundaryYearRepresented"))
