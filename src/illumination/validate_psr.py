import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt

TILE = "data/Site04_final_adj_5mpp_surf.tif"
PSR  = "data/LPSR_80S_20MPP_ADJ.tiff"

with rasterio.open(TILE) as t:
    tb = t.bounds

with rasterio.open(PSR) as p:
    win = from_bounds(tb.left, tb.bottom, tb.right, tb.top, p.transform)
    psr = p.read(1, window=win)

print("PSR crop shape:", psr.shape)
print("unique values:", np.unique(psr[np.isfinite(psr)])[:10])
print("nan count:", int(np.isnan(psr).sum()))

finite = psr[np.isfinite(psr)]
print("min:", float(finite.min()), "max:", float(finite.max()))
print("fraction non-zero:", float((finite > 0).mean()))

plt.figure(figsize=(7, 7))
plt.imshow(psr, cmap="magma")
plt.colorbar(shrink=0.8)
plt.title("NASA PSR map, Site04 footprint")
plt.xticks([]); plt.yticks([])
plt.tight_layout()
plt.savefig("figures/psr_reference.png", dpi=150)
print("saved figures/psr_reference.png")