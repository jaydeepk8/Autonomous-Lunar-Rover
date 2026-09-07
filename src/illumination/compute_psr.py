import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
from numba import njit, prange
import time

TILE = "data/Site04_final_adj_5mpp_surf.tif"
WIDE = "data/LDEM_80S_40MPP_ADJ.tiff"
PSR  = "data/LPSR_80S_20MPP_ADJ.tiff"

MARGIN_M = 30000.0      # how far beyond the tile rays may travel
MAX_RANGE_M = 30000.0
N_AZIMUTHS = 72         # 10-degree steps
SUN_ELEV_MAX = 1.9     # highest the sun ever gets at this site


@njit(parallel=True, cache=True)
def never_lit(dem, res, r0, r1, c0, c1, elev_deg, n_az, max_range_m):
    """
    For the sub-window [r0:r1, c0:c1] of dem, return 1 where the cell is
    shadowed at EVERY azimuth (a PSR), 0 otherwise.
    """
    h, w = dem.shape
    out = np.ones((r1 - r0, c1 - c0), dtype=np.uint8)
    n_steps = int(max_range_m / res)

    for row in prange(r0, r1):
        for col in range(c0, c1):
            z0 = dem[row, col]
            if not np.isfinite(z0):
                out[row - r0, col - c0] = 0
                continue

            lit_somewhere = False
            for k in range(n_az):
                a = 2.0 * np.pi * k / n_az
                dr = np.cos(a)
                dc = np.sin(a)

                max_angle = -90.0
                for i in range(1, n_steps + 1):
                    r = int(round(row + dr * i))
                    c = int(round(col + dc * i))
                    if r < 0 or r >= h or c < 0 or c >= w:
                        break
                    z = dem[r, c]
                    if not np.isfinite(z):
                        continue
                    angle = np.degrees(np.arctan2(z - z0, i * res))
                    if angle > max_angle:
                        max_angle = angle

                if elev_deg > max_angle:
                    lit_somewhere = True
                    break

            if lit_somewhere:
                out[row - r0, col - c0] = 0

    return out


if __name__ == "__main__":
    with rasterio.open(TILE) as t:
        tb = t.bounds

    with rasterio.open(WIDE) as w:
        win = from_bounds(tb.left - MARGIN_M, tb.bottom - MARGIN_M,
                          tb.right + MARGIN_M, tb.top + MARGIN_M,
                          w.transform)
        dem = w.read(1, window=win).astype(np.float64)
        wt = w.window_transform(win)
        res = w.res[0]

    print("working DEM:", dem.shape, "at", res, "m/px")

    r0, c0 = rowcol(wt, tb.left, tb.top)
    r1, c1 = rowcol(wt, tb.right, tb.bottom)
    print("Site04 footprint inside it: rows", r0, r1, "cols", c0, c1)

    t0 = time.time()
    mine = never_lit(dem, res, r0, r1, c0, c1,
                     SUN_ELEV_MAX, N_AZIMUTHS, MAX_RANGE_M)
    print(f"computed in {time.time() - t0:.1f} s")

    with rasterio.open(PSR) as p:
        pwin = from_bounds(tb.left, tb.bottom, tb.right, tb.top, p.transform)
        ref = p.read(1, window=pwin)
    factor = ref.shape[0] // mine.shape[0]
    ref = ref[::factor, ::factor][:mine.shape[0], :mine.shape[1]]

    m = mine.astype(bool)
    r = ref.astype(bool)

    agree = (m == r).mean()
    inter = (m & r).sum()
    union = (m | r).sum()
    print(f"\nmy PSR fraction:   {m.mean():.4f}")
    print(f"NASA PSR fraction: {r.mean():.4f}")
    print(f"cell agreement:    {agree:.4f}")
    print(f"IoU:               {inter / union:.4f}")
    print(f"missed (NASA yes, mine no): {(r & ~m).sum()}")
    print(f"extra  (mine yes, NASA no): {(m & ~r).sum()}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.5))
    ax[0].imshow(r, cmap="gray"); ax[0].set_title("NASA PSR")
    ax[1].imshow(m, cmap="gray"); ax[1].set_title("Computed PSR")
    diff = np.zeros(m.shape + (3,))
    diff[..., 0] = m & ~r      # red   = mine only
    diff[..., 1] = m & r       # green = both
    diff[..., 2] = r & ~m      # blue  = NASA only
    ax[2].imshow(diff); ax[2].set_title("green=both  red=mine  blue=NASA")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    plt.tight_layout()
    plt.savefig("figures/psr_validation.png", dpi=150)
    print("saved figures/psr_validation.png")