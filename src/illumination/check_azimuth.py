import sys
sys.path.insert(0, "src/illumination")

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import transform as warp_transform

from sun_geometry import sun_position, SYNODIC_HOURS
from illumination_stack import lit_mask, TILE, WIDE

PROBE_M = 50000.0
MOON_R = 1737400.0


def geographic_of(crs):
    d = crs.to_dict()
    g = {"proj": "longlat", "no_defs": True}
    for k in ("R", "a", "b", "ellps", "datum"):
        if k in d:
            g[k] = d[k]
    if not any(k in g for k in ("R", "a", "ellps", "datum")):
        g["R"] = MOON_R
    return CRS.from_dict(g)


def lon_of(crs, xs, ys):
    try:
        lons, _ = warp_transform(crs, geographic_of(crs), list(xs), list(ys))
        return np.array(lons), "PROJ"
    except Exception as e:
        d = crs.to_dict()
        if d.get("proj") == "stere" and float(d.get("lat_0", 0)) == -90.0:
            # south polar stereographic: x = r sin(lon - lon_0), y = r cos(lon - lon_0)
            lon0 = float(d.get("lon_0", 0.0))
            return np.degrees(np.arctan2(xs, ys)) + lon0, "south-polar formula"
        raise RuntimeError(f"cannot convert to longitude: {e}\nCRS: {crs}")


def part1_sun_sense():
    print("=" * 70)
    print("PART 1  which way the modelled sun azimuth moves with time")
    print("=" * 70)
    t = np.linspace(0, SYNODIC_HOURS, 721)
    elev, azim = sun_position(t)
    total = np.degrees(np.unwrap(np.radians(azim)))
    total = total[-1] - total[0]
    print(f"  elevation {elev.min():.2f} to {elev.max():.2f} deg")
    print(f"  azimuth change over the month: {total:+.1f} deg")
    sense = 1 if total > 0 else -1
    print(f"  -> azimuth {'increases' if sense > 0 else 'decreases'} with time\n")
    return sense


def part2_lon_sense(crs, transform, x0, y0):
    print("=" * 70)
    print("PART 2  which way longitude runs as array azimuth increases")
    print("=" * 70)
    print(f"  CRS: {crs.to_string()}")
    if transform.b != 0 or transform.d != 0 or transform.e >= 0:
        print("  !! raster is rotated or not north-up; the array convention below")
        print("     assumes +row = -y and +col = +x. Stop and check.")
    az = np.arange(0.0, 360.0, 5.0)
    a = np.radians(az)
    xs = x0 + np.sin(a) * PROBE_M
    ys = y0 - np.cos(a) * PROBE_M
    lons, how = lon_of(crs, xs, ys)
    total = np.degrees(np.unwrap(np.radians(lons)))
    total = total[-1] - total[0]
    print(f"  longitude via {how}")
    print(f"  longitude change over one turn of array azimuth: {total:+.1f} deg")
    if abs(abs(total) - 355.0) > 20.0:
        print("  !! not close to a full turn: the probe circle did not enclose")
        print("     the pole. Increase PROBE_M.")
    sense = 1 if total > 0 else -1
    print(f"  -> longitude {'increases' if sense > 0 else 'decreases'} "
          f"as array azimuth increases\n")
    return sense


def part3_verdict(sun_sense, lon_sense):
    print("=" * 70)
    print("PART 3  verdict")
    print("=" * 70)
    modelled = sun_sense * lon_sense
    print(f"  modelled sun moves toward {'EAST' if modelled > 0 else 'WEST'}")
    print("  real sun moves toward      WEST  (subsolar longitude decreases)\n")
    if modelled < 0:
        print("  PASS  the sun goes round the right way. Nothing to change.")
    else:
        print("  FAIL  the sun goes round backwards: the month plays in reverse.")
        print("        Fix is in sun_geometry.py (sign of the hour angle).")
        print("        Unaffected: PSR map and IoU, horizon field, month-mean")
        print("        power, longest dark stretch, sunrises per month.")
        print("        Affected: anything that depends on the order of events.")
    print()


def part4_shadow_direction():
    print("=" * 70)
    print("PART 4  shadows fall away from the sun (synthetic cone)")
    print("=" * 70)
    n, res = 201, 40.0
    c = n // 2
    rr, cc = np.mgrid[0:n, 0:n]
    dem = np.maximum(0.0, 600.0 - np.hypot(rr - c, cc - c) * res * 0.6)
    ok = True
    for az in (0.0, 90.0, 180.0, 270.0):
        m = lit_mask(dem, res, 0, n, 0, n, 5.0, az, 20000.0)
        sh = (m == 0) & (dem < 1.0)
        sr, sc = rr[sh].mean() - c, cc[sh].mean() - c
        nrm = np.hypot(sr, sc)
        dot = (sr * -np.cos(np.radians(az)) + sc * -np.sin(np.radians(az))) / nrm
        good = dot > 0.8
        ok &= good
        print(f"  sun azimuth {az:5.1f}: shadow alignment {dot:+.2f}  "
              f"{'OK' if good else 'MISMATCH'}")
    print(f"\n  {'all OK' if ok else 'ray direction is wrong'}\n")


if __name__ == "__main__":
    with rasterio.open(TILE) as t:
        tb = t.bounds
    with rasterio.open(WIDE) as w:
        crs, transform = w.crs, w.transform
    x0 = (tb.left + tb.right) / 2.0
    y0 = (tb.bottom + tb.top) / 2.0

    s1 = part1_sun_sense()
    s2 = part2_lon_sense(crs, transform, x0, y0)
    part3_verdict(s1, s2)
    part4_shadow_direction()