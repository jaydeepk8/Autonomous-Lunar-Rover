import numpy as np
import matplotlib.pyplot as plt

SITE_LAT_DEG = -89.9

OBLIQUITY_DEG = 1.54

SYNODIC_HOURS = 29.53 * 24


def sun_position(hours, lat_deg=SITE_LAT_DEG, subsolar_lat_deg=-1.54):
    """
    Sun position at a lunar polar site over one synodic month.

    subsolar_lat_deg is the SEASONAL term: it varies on annual and
    18.6-year timescales, not within a month. Negative values put the
    south pole in polar day. Hold it fixed for a one-month stack.
    """
    lat = np.radians(lat_deg)
    dec = np.radians(subsolar_lat_deg)
    H = 2 * np.pi * hours / SYNODIC_HOURS          # hour angle

    sin_h = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(H)
    elevation = np.degrees(np.arcsin(sin_h))

    azimuth = np.degrees(np.arctan2(
        -np.sin(H) * np.cos(dec),
        np.cos(lat) * np.sin(dec) - np.sin(lat) * np.cos(dec) * np.cos(H)
    )) % 360.0

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
    ax[0].set_ylabel("sun elevation (deg)")
    ax[0].set_title("Elevation over one synodic month")

    ax[1].plot(hours / 24, azim)
    ax[1].set_xlabel("days")
    ax[1].set_ylabel("sun azimuth (deg)")
    ax[1].set_title("Azimuth over one synodic month")

    plt.tight_layout()
    plt.savefig("figures/sun_geometry.png", dpi=150)
    print("saved figures/sun_geometry.png")
