import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

PATH = "data/Site04_final_adj_5mpp_surf.tif"

with rasterio.open(PATH) as src:
    dem = src.read(1)
    res = src.res[0]

filled = np.where(np.isfinite(dem), dem, np.nanmin(dem))

ls = LightSource(azdeg=315, altdeg=45)
shade = ls.hillshade(filled, vert_exag=1.0, dx=res, dy=res)

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

im = ax[0].imshow(dem, cmap="viridis")
ax[0].set_title("Elevation (m)")
plt.colorbar(im, ax=ax[0], shrink=0.7)

ax[1].imshow(shade, cmap="gray")
ax[1].set_title("Hillshade")

for a in ax:
    a.set_xticks([])
    a.set_yticks([])

plt.tight_layout()
plt.savefig("figures/site04_terrain.png", dpi=150)
print("saved figures/site04_terrain.png")