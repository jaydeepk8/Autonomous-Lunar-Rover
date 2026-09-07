import numpy as np
import rasterio
import matplotlib.pyplot as plt
from numba import njit, prange

PATH = "data/Site04_final_adj_5mpp_surf.tif"
DOWNSAMPLE = 4          # 5 m -> 20 m
MAX_RANGE_M = 20000.0


@njit(parallel=True, cache=True)
def shadow_mask(dem, res, sun_elev_deg, sun_azim_deg, max_range_m):
    """
    1 where the cell is lit, 0 where shadowed, for one sun position.
    """
    h, w = dem.shape
    out = np.zeros((h, w), dtype=np.uint8)

    a = np.radians(sun_azim_deg)
    dr = np.cos(a)
    dc = np.sin(a)
    n_steps = int(max_range_m / res)

    for row in prange(h):
        for col in range(w):
            z0 = dem[row, col]
            if not np.isfinite(z0):
                continue

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

            if sun_elev_deg > max_angle:
                out[row, col] = 1

    return out


if __name__ == "__main__":
    with rasterio.open(PATH) as src:
        dem_full = src.read(1)
        res_full = src.res[0]

    dem = dem_full[::DOWNSAMPLE, ::DOWNSAMPLE].astype(np.float64)
    res = res_full * DOWNSAMPLE
    print("working grid:", dem.shape, "at", res, "m/px")

    SUN_ELEV = 1.5
    SUN_AZIM = 270.0

    import time
    t0 = time.time()
    mask = shadow_mask(dem, res, SUN_ELEV, SUN_AZIM, MAX_RANGE_M)
    print(f"computed in {time.time() - t0:.1f} s")

    lit_frac = mask.mean()
    print(f"lit fraction: {lit_frac:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].imshow(dem, cmap="viridis")
    ax[0].set_title("Elevation (m)")
    ax[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
    ax[1].set_title(f"Lit (white) at elev {SUN_ELEV}, azim {SUN_AZIM}")
    for a_ in ax:
        a_.set_xticks([]); a_.set_yticks([])
    plt.tight_layout()
    plt.savefig("figures/shadow_mask.png", dpi=150)
    print("saved figures/shadow_mask.png")