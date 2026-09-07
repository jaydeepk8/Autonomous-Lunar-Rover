import numpy as np
import matplotlib.pyplot as plt

SITE_LAT_DEG = -89.9

OBLIQUITY_DEG = 1.54

SYNODIC_HOURS = 29.53 * 24


def sun_position(hours, lat_deg=SITE_LAT_DEG, obliquity_deg=OBLIQUITY_DEG):
    """
    Analytic sun position at a lunar polar site.
    Returns (elevation_deg, azimuth_deg).

    Approximation: sub-solar latitude oscillates with the obliquity over
    one synodic month; azimuth sweeps a full 360 degrees in the same period.
    """
    lat = np.radians(lat_deg)
    phase = 2 * np.pi * hours / SYNODIC_HOURS

    subsolar_lat = np.radians(obliquity_deg) * np.sin(phase)

    elevation = np.degrees(subsolar_lat - lat) - 90.0
    
    azimuth = np.degrees(phase) % 360.0

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
