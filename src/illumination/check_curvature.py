import sys
sys.path.insert(0, "src/illumination")

import time
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import rowcol
from rasterio.warp import transform as warp_transform
from rasterio.crs import CRS
from numba import njit, prange

from compute_psr import never_lit, TILE, WIDE, PSR, MARGIN_M, MAX_RANGE_M, N_AZIMUTHS

R_MOON = 1737400.0
SUN_DEC_MAX = 1.54
DISC = 0.26
FLAT_ELEV = 1.9


@njit(parallel=True, cache=True)
def never_lit_curved(dem, res, r0, r1, c0, c1, n_az, max_range_m,
                     base_elev, xs, ys, per_cell):
    h, w = dem.shape
    out = np.ones((r1 - r0, c1 - c0), dtype=np.uint8)
    n_steps = int(max_range_m / res)
    for row in prange(r0, r1):
        for col in range(c0, c1):
            z0 = dem[row, col]
            if not np.isfinite(z0):
                out[row - r0, col - c0] = 0
                continue
            for k in range(n_az):
                a = 2.0 * np.pi * k / n_az
                dr = np.cos(a)
                dc = np.sin(a)
                elev = base_elev
                if per_cell:
                    # sun sits higher for cells displaced from the pole toward it
                    elev += np.degrees((xs[col - c0] * dc - ys[row - r0] * dr) / R_MOON)
                max_angle = -90.0
                for i in range(1, n_steps + 1):
                    r = int(round(row + dr * i))
                    c = int(round(col + dc * i))
                    if r < 0 or r >= h or c < 0 or c >= w:
                        break
                    z = dem[r, c]
                    if not np.isfinite(z):
                        continue
                    d = i * res
                    angle = np.degrees(np.arctan2(z - z0 - d * d / (2.0 * R_MOON), d))
                    if angle > max_angle:
                        max_angle = angle
                        if max_angle >= elev:
                            break
                if elev > max_angle:
                    out[row - r0, col - c0] = 0
                    break
    return out


def score(mine, ref):
    m, r = mine.astype(bool), ref.astype(bool)
    return (m & r).sum() / (m | r).sum(), m.mean()


def site_geometry(crs, xs, ys):
    xc, yc = xs.mean(), ys.mean()
    d = crs.to_dict()
    g = {"proj": "longlat", "R": d.get("R", R_MOON), "no_defs": True}
    try:
        lon, lat = warp_transform(crs, CRS.from_dict(g), [xc], [yc])
        lon, lat = lon[0], lat[0]
    except Exception:
        lon = np.degrees(np.arctan2(xc, yc))
        lat = -90.0 + np.degrees(np.hypot(xc, yc) / R_MOON)
    s = np.hypot.outer(ys, xs)
    pole_inside = xs.min() <= 0 <= xs.max() and ys.min() <= 0 <= ys.max()
    print(f"  site centre        : lat {lat:.3f}, lon {lon:.1f}")
    print(f"  centre to pole     : {np.hypot(xc, yc) / 1000:.1f} km")
    print(f"  cells to pole      : {s.min() / 1000:.1f} to {s.max() / 1000:.1f} km")
    print(f"  pole inside site   : {'yes' if pole_inside else 'no'}")
    print(f"  sun_geometry.py assumes lat -89.9, which is "
          f"{0.1 * np.pi / 180 * R_MOON / 1000:.1f} km from the pole\n")


if __name__ == "__main__":
    with rasterio.open(TILE) as t:
        tb = t.bounds
    with rasterio.open(WIDE) as w:
        win = from_bounds(tb.left - MARGIN_M, tb.bottom - MARGIN_M,
                          tb.right + MARGIN_M, tb.top + MARGIN_M, w.transform)
        dem = w.read(1, window=win).astype(np.float64)
        wt = w.window_transform(win)
        res = w.res[0]
        crs = w.crs

    r0, c0 = rowcol(wt, tb.left, tb.top)
    r1, c1 = rowcol(wt, tb.right, tb.bottom)
    xs = wt.c + (np.arange(c0, c1) + 0.5) * wt.a
    ys = wt.f + (np.arange(r0, r1) + 0.5) * wt.e

    with rasterio.open(PSR) as p:
        ref = p.read(1, window=from_bounds(tb.left, tb.bottom, tb.right, tb.top,
                                           p.transform))
    f = ref.shape[0] // (r1 - r0)
    ref = ref[::f, ::f][:r1 - r0, :c1 - c0]

    print("=" * 70)
    print("SITE GEOMETRY")
    print("=" * 70)
    site_geometry(crs, xs, ys)

    print("=" * 70)
    print("PSR MAP AGAINST NASA, THREE WAYS")
    print("=" * 70)
    runs = [
        ("1 flat ground, sun 1.9 deg everywhere (current)",
         lambda: never_lit(dem, res, r0, r1, c0, c1, FLAT_ELEV, N_AZIMUTHS, MAX_RANGE_M)),
        ("2 curved ground, sun 1.9 deg everywhere",
         lambda: never_lit_curved(dem, res, r0, r1, c0, c1, N_AZIMUTHS, MAX_RANGE_M,
                                  FLAT_ELEV, xs, ys, False)),
        ("3 curved ground, sun height per cell",
         lambda: never_lit_curved(dem, res, r0, r1, c0, c1, N_AZIMUTHS, MAX_RANGE_M,
                                  SUN_DEC_MAX + DISC, xs, ys, True)),
    ]
    maps = []
    for name, fn in runs:
        t0 = time.time()
        m = fn()
        iou, frac = score(m, ref)
        maps.append(m)
        print(f"  {name}")
        print(f"      IoU {iou:.4f}   PSR fraction {frac:.4f}   ({time.time() - t0:.0f} s)")
    print(f"\n  NASA PSR fraction {ref.astype(bool).mean():.4f}")
    print(f"  cells that change between 1 and 3: {(maps[0] != maps[2]).mean():.2%}")