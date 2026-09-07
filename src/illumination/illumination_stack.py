import sys
sys.path.insert(0, "src/illumination")

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
import matplotlib.pyplot as plt
from numba import njit, prange
import time as clock

from sun_geometry import sun_position, SYNODIC_HOURS

TILE = "data/Site04_final_adj_5mpp_surf.tif"
WIDE = "data/LDEM_80S_40MPP_ADJ.tiff"

MARGIN_M = 30000.0
MAX_RANGE_M = 30000.0
N_TIMES = 72                 # ~10-hour steps over one synodic month
DISC_OFFSET = 0.26           # solar upper limb, from the validation


@njit(parallel=True, cache=True)
def lit_mask(dem, res, r0, r1, c0, c1, elev_deg, azim_deg, max_range_m):
    """1 where lit, 0 where shadowed, for one sun position."""
    h, w = dem.shape
    out = np.zeros((r1 - r0, c1 - c0), dtype=np.uint8)

    if elev_deg <= 0.0:
        return out

    a = np.radians(azim_deg)
    dr = np.cos(a)
    dc = np.sin(a)
    n_steps = int(max_range_m / res)

    for row in prange(r0, r1):
        for col in range(c0, c1):
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

            if elev_deg > max_angle:
                out[row - r0, col - c0] = 1

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

    r0, c0 = rowcol(wt, tb.left, tb.top)
    r1, c1 = rowcol(wt, tb.right, tb.bottom)
    H, W = r1 - r0, c1 - c0
    print("footprint:", H, "x", W, "at", res, "m/px")

    hours = np.linspace(0, SYNODIC_HOURS, N_TIMES, endpoint=False)
    stack = np.zeros((N_TIMES, H, W), dtype=np.uint8)

    t0 = clock.time()
    for k, hr in enumerate(hours):
        elev, azim = sun_position(hr)
        stack[k] = lit_mask(dem, res, r0, r1, c0, c1,
                            float(elev) + DISC_OFFSET, float(azim),
                            MAX_RANGE_M)
        if k % 12 == 0:
            print(f"  t={hr/24:5.1f} d  elev={elev:6.2f}  "
                  f"azim={azim:6.1f}  lit={stack[k].mean():.3f}")
    print(f"stack computed in {clock.time() - t0:.1f} s")

    np.save("results/illumination_stack.npy", stack)
    np.save("results/illumination_hours.npy", hours)
    print("saved results/illumination_stack.npy", stack.shape,
          f"({stack.nbytes/1e6:.1f} MB)")

    frac = stack.mean(axis=0)
    print("\nmean illumination fraction:", float(frac.mean()))
    print("cells never lit (PSR):", int((frac == 0).sum()),
          f"({(frac == 0).mean():.3f})")
    print("cells lit >80% of month:", int((frac > 0.8).sum()))
    print("max illumination fraction:", float(frac.max()))

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    im = ax[0].imshow(frac, cmap="inferno", vmin=0, vmax=1)
    ax[0].set_title("Illumination fraction over one synodic month")
    plt.colorbar(im, ax=ax[0], shrink=0.75)
    ax[0].set_xticks([]); ax[0].set_yticks([])

    ax[1].plot(hours / 24, stack.reshape(N_TIMES, -1).mean(axis=1))
    ax[1].set_xlabel("days")
    ax[1].set_ylabel("lit fraction of site")
    ax[1].set_title("Site-wide illumination over time")

    plt.tight_layout()
    plt.savefig("figures/illumination_fraction.png", dpi=150)
    print("saved figures/illumination_fraction.png")