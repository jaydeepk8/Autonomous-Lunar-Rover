import numpy as np
import rasterio

PATH = "data/Site04_final_adj_5mpp_surf.tif"


def horizon_angle(dem, res, row, col, azimuth_deg, max_range_m=20000.0):
    """
    Maximum elevation angle to the horizon from (row, col), looking along
    azimuth_deg. Returns degrees above the local horizontal.

    Azimuth convention: 0 = +row direction (down the array), increasing
    clockwise. This is arbitrary for now; it gets pinned to the map
    projection in the next step.
    """
    h, w = dem.shape
    z0 = dem[row, col]
    if not np.isfinite(z0):
        return np.nan

    a = np.radians(azimuth_deg)
    dr, dc = np.cos(a), np.sin(a)

    n_steps = int(max_range_m / res)
    max_angle = -90.0

    for i in range(1, n_steps + 1):
        r = int(round(row + dr * i))
        c = int(round(col + dc * i))
        if r < 0 or r >= h or c < 0 or c >= w:
            break
        z = dem[r, c]
        if not np.isfinite(z):
            continue
        dist = i * res
        angle = np.degrees(np.arctan2(z - z0, dist))
        if angle > max_angle:
            max_angle = angle

    return max_angle


if __name__ == "__main__":
    with rasterio.open(PATH) as src:
        dem = src.read(1)
        res = src.res[0]

    rim = np.unravel_index(np.nanargmax(dem), dem.shape)
    floor = np.unravel_index(np.nanargmin(dem), dem.shape)

    print("rim cell:", rim, "elevation:", float(dem[rim]))
    print("floor cell:", floor, "elevation:", float(dem[floor]))

    for name, cell in [("rim", rim), ("floor", floor)]:
        angles = [horizon_angle(dem, res, cell[0], cell[1], az)
                  for az in range(0, 360, 30)]
        print(f"\n{name} horizon angles (deg), azimuth 0-330 step 30:")
        print(np.round(angles, 2))
        print(f"{name} max horizon angle: {np.nanmax(angles):.2f}")