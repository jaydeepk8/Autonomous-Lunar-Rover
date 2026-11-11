import numpy as np
import matplotlib.pyplot as plt

# reference point is the pole; each cell's own sun height is added in
# horizon_field.sun_elevation()
SITE_LAT_DEG = -90.0

OBLIQUITY_DEG = 1.54

SYNODIC_HOURS = 29.53 * 24

# season: subsolar latitude, held fixed within a month; -1.54 is peak south
# polar summer. Sweeps change this value instead of passing it around.
SEASON = {"subsolar_lat_deg": -1.54}


def sun_position(hours, lat_deg=SITE_LAT_DEG, subsolar_lat_deg=None):
    """
    Sun position at a lunar polar site over one synodic month.

    subsolar_lat_deg is the SEASONAL term: it varies over a ~346.6-day
    lunar seasonal cycle, not within a month. Negative values put the
    south pole in polar day. Defaults to SEASON["subsolar_lat_deg"].
    """
    if subsolar_lat_deg is None:
        subsolar_lat_deg = SEASON["subsolar_lat_deg"]
    lat = np.radians(lat_deg)
    dec = np.radians(subsolar_lat_deg)
    H = 2 * np.pi * hours / SYNODIC_HOURS          # hour angle

    sin_h = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H)
    elevation = np.degrees(np.arcsin(sin_h))

    compass = np.degrees(np.arctan2(
        -np.sin(H) * np.cos(dec),
        np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(H)
    ))
    # map-grid azimuth runs the opposite way round from compass bearing at the
    # south pole (checked by check_azimuth.py)
    azimuth = (-compass) % 360.0

    return elevation, azimuth


if __name__ == "__main__":
    hours = np.linspace(0, SYNODIC_HOURS, 720)
    elev, azim = sun_position(hours)

    print("elevation min (deg):", float(elev.min()))
    print("elevation max (deg):", float(elev.max()))
    print("hours per full azimuth sweep:", SYNODIC_HOURS)

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    ax[0].plot(hours / 24, elev)
    ax[0].axhline(0, color="k", lw=0.8, ls="--")
    ax[0].set_xlabel("days")
    ax[0].set_ylabel("sun elevation at the pole (deg)")
    ax[0].set_title("Elevation over one synodic month")

    ax[1].plot(hours / 24, azim)
    ax[1].set_xlabel("days")
    ax[1].set_ylabel("sun azimuth, map grid (deg)")
    ax[1].set_title("Azimuth over one synodic month")

    plt.tight_layout()
    plt.savefig("figures/sun_geometry.png", dpi=150)
    print("saved figures/sun_geometry.png")