import sys
sys.path.insert(0, "src/illumination")

import time as clock
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from numba import njit, prange

from sun_geometry import sun_position, SYNODIC_HOURS

TILE = "data/Site04_final_adj_5mpp_surf.tif"
WIDE = "data/LDEM_80S_40MPP_ADJ.tiff"
PSR = "data/LPSR_80S_20MPP_ADJ.tiff"

OUT = "results/horizon_field.npy"

MARGIN_M = 30000.0
MAX_RANGE_M = 30000.0
N_AZ = 72
DISC_RADIUS_DEG = 0.26
SUN_ELEV_MAX = 1.9


@njit(parallel=True, cache=True)
def horizon_field(dem, res, r0, r1, c0, c1, n_az, max_range_m):
    out = np.empty((r1 - r0, c1 - c0, n_az), dtype=np.float32)
    n_steps = int(max_range_m / res)

    for row in prange(r0, r1):
        for col in range(c0, c1):
            z0 = dem[row, col]
            if not np.isfinite(z0):
                for k in range(n_az):
                    out[row - r0, col - c0, k] = 90.0
                continue

            for k in range(n_az):
                a = 2.0 * np.pi * k / n_az
                dr = np.cos(a)
                dc = np.sin(a)

                max_angle = -90.0
                for i in range(1, n_steps + 1):
                    r = int(round(row + dr * i))
                    c = int(round(col + dc * i))
                    if r < 0 or r >= dem.shape[0] or c < 0 or c >= dem.shape[1]:
                        break
                    z = dem[r, c]
                    if not np.isfinite(z):
                        continue
                    angle = np.degrees(np.arctan2(z - z0, i * res))
                    if angle > max_angle:
                        max_angle = angle

                out[row - r0, col - c0, k] = max_angle

    return out


def horizon_at(H, azim_deg):
    n_az = H.shape[2]
    x = (azim_deg % 360.0) / 360.0 * n_az
    k0 = int(np.floor(x)) % n_az
    k1 = (k0 + 1) % n_az
    w = x - np.floor(x)
    return (1.0 - w) * H[:, :, k0] + w * H[:, :, k1]


def disc_fraction(H, elev_deg, azim_deg, disc_radius=DISC_RADIUS_DEG):
    h = horizon_at(H, azim_deg)
    f = (elev_deg + disc_radius - h) / (2.0 * disc_radius)
    return np.clip(f, 0.0, 1.0)


def psr_from_field(H, elev_max=SUN_ELEV_MAX):
    return (elev_max <= H.min(axis=2)).astype(np.uint8)


def load_dem():
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
    return dem, res, r0, r1, c0, c1, tb


def validate(H, tb):
    mine = psr_from_field(H)

    with rasterio.open(PSR) as p:
        pwin = from_bounds(tb.left, tb.bottom, tb.right, tb.top, p.transform)
        ref = p.read(1, window=pwin)
    factor = ref.shape[0] // mine.shape[0]
    ref = ref[::factor, ::factor][:mine.shape[0], :mine.shape[1]]

    m = mine.astype(bool)
    r = ref.astype(bool)
    iou = (m & r).sum() / (m | r).sum()

    print(f"  my PSR fraction   : {m.mean():.4f}")
    print(f"  NASA PSR fraction : {r.mean():.4f}")
    print(f"  IoU               : {iou:.4f}")
    print(f"  Phase 0 reference : 0.9675")
    if abs(iou - 0.9675) < 0.01:
        print("  MATCH -- the field reproduces the validated Phase 0 result")
    else:
        print("  MISMATCH -- do not proceed, the field disagrees with Phase 0")
    return iou


if __name__ == "__main__":
    dem, res, r0, r1, c0, c1, tb = load_dem()
    H_cells, W_cells = r1 - r0, c1 - c0
    nbytes = H_cells * W_cells * N_AZ * 4

    print(f"footprint {H_cells} x {W_cells} at {res} m/px")
    print(f"{N_AZ} azimuths -> {nbytes / 1e6:.0f} MB float32")
    print("this is a one-time compute and will take a while\n")

    t0 = clock.time()
    H = horizon_field(dem, res, r0, r1, c0, c1, N_AZ, MAX_RANGE_M)
    print(f"computed in {clock.time() - t0:.1f} s\n")

    np.save(OUT, H)
    print(f"saved {OUT}\n")

    print("VALIDATION against NASA PSR map")
    validate(H, tb)

    print("\nILLUMINATION AT ARBITRARY TIME (no recompute)")
    for hr in [0.0, 1.0, 6.0, 100.0, 350.0]:
        elev, azim = sun_position(hr)
        f = disc_fraction(H, float(elev), float(azim))
        print(f"  t = {hr:7.2f} h  elev {elev:5.2f}  azim {azim:6.1f}  "
              f"mean f {f.mean():.4f}  fully lit {(f >= 1.0).mean():.4f}  "
              f"partial {((f > 0) & (f < 1)).mean():.4f}")

    print(f"\nold stack: 72 frames, one every {SYNODIC_HOURS / 72:.2f} h")
    print("this field: any time, any resolution, by lookup")