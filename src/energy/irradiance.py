import sys
sys.path.insert(0, "src/illumination")

import numpy as np

from sun_geometry import sun_position, SYNODIC_HOURS
from horizon_field import horizon_at, DISC_RADIUS_DEG

FIELD = "results/horizon_field.npy"

PARAMS = {
    "P_array_peak": dict(
        value=450.0, unit="W", tag="spec",
        source="Bluethmann, NTRS 20240013903: 320 W per panel, 450 W on corner"),
    "solar_constant": dict(
        value=1361.0, unit="W/m2", tag="spec",
        source="solar irradiance at 1 AU; Moon-Sun distance varies it by +-3.4 % "
               "over a year, not modelled"),
    "panel": dict(
        value="vertical_tracking", unit="-", tag="assumed",
        source="VIPER drives in any direction without changing facing, to keep "
               "panels pointed at the Sun"),
}


def cos_incidence(elev_deg, panel="vertical_tracking", tilt_deg=0.0):
    e = np.radians(elev_deg)
    t = np.radians(tilt_deg)
    if panel == "vertical_tracking":
        return np.maximum(np.cos(e - t), 0.0)
    if panel == "horizontal":
        return np.maximum(np.sin(e + t), 0.0)
    raise ValueError(panel)


def solar_power(H, elev_deg, azim_deg, panel="vertical_tracking",
                tilt_deg=0.0, P_peak=None):
    if P_peak is None:
        P_peak = PARAMS["P_array_peak"]["value"]
    h = horizon_at(H, azim_deg)
    f = np.clip((elev_deg + DISC_RADIUS_DEG - h) / (2.0 * DISC_RADIUS_DEG), 0.0, 1.0)
    return P_peak * f * cos_incidence(elev_deg, panel, tilt_deg)


def month_mean_power(H, n_steps=720, panel="vertical_tracking", tilt_deg=0.0):
    acc = np.zeros(H.shape[:2], dtype=np.float64)
    for hr in np.linspace(0.0, SYNODIC_HOURS, n_steps, endpoint=False):
        elev, azim = sun_position(hr)
        acc += solar_power(H, float(elev), float(azim), panel, tilt_deg)
    return acc / n_steps


def report_panel_geometry():
    print("=" * 74)
    print("PANEL GEOMETRY  (why mounting dominates everything)")
    print("=" * 74)
    print(f"  {'sun elev':>9s} {'vertical':>10s} {'horizontal':>11s} {'ratio':>8s}")
    for e in [1.44, 1.50, 1.64, 5.0, 15.0]:
        v = float(cos_incidence(e, "vertical_tracking"))
        h = float(cos_incidence(e, "horizontal"))
        print(f"  {e:9.2f} {v:10.4f} {h:11.4f} {v / h:8.1f}")
    print()


def report_tilt_sensitivity():
    print("=" * 74)
    print("TERRAIN TILT SENSITIVITY  (sun at 1.54 deg)")
    print("=" * 74)
    e = 1.54
    print(f"  {'tilt':>6s} {'vertical':>10s} {'horizontal':>11s}")
    for t in [0.0, 5.0, 10.0, 15.0, 25.0]:
        v = float(cos_incidence(e, "vertical_tracking", t))
        h = float(cos_incidence(e, "horizontal", t))
        print(f"  {t:6.1f} {v:10.4f} {h:11.4f}")
    print()


def report_instants(H):
    print("=" * 74)
    print("SOLAR POWER AT A FEW INSTANTS")
    print("=" * 74)
    print(f"  {'t (h)':>8s} {'elev':>6s} {'azim':>7s} {'mean W':>9s} "
          f"{'max W':>8s} {'cells > 100 W':>14s}")
    for hr in [0.0, 100.0, 200.0, 350.0, 500.0]:
        elev, azim = sun_position(hr)
        P = solar_power(H, float(elev), float(azim))
        print(f"  {hr:8.1f} {elev:6.2f} {azim:7.1f} {P.mean():9.2f} "
              f"{P.max():8.1f} {(P > 100).mean():14.4f}")
    print()


def report_month(H):
    print("=" * 74)
    print("MONTH-AVERAGED POWER  (720 samples, one per hour)")
    print("=" * 74)
    Pm = month_mean_power(H)
    print(f"  site mean        : {Pm.mean():7.2f} W")
    print(f"  best cell        : {Pm.max():7.2f} W")
    print(f"  cells above 50 W : {(Pm > 50).sum():7d}  ({(Pm > 50).mean():.4f})")
    print(f"  cells above 100 W: {(Pm > 100).sum():7d}  ({(Pm > 100).mean():.4f})")
    print(f"  cells at zero    : {(Pm == 0).sum():7d}  ({(Pm == 0).mean():.4f})")
    print()
    print(f"  best cell receives {Pm.max() / PARAMS['P_array_peak']['value'] * 100:.1f} % "
          f"of array peak, averaged over the month")
    print()
    return Pm


if __name__ == "__main__":
    H = np.load(FIELD)
    print(f"\nhorizon field {H.shape} {H.dtype}\n")

    print("=" * 74)
    print("PARAMETERS")
    print("=" * 74)
    for k, d in PARAMS.items():
        print(f"  {k:16s} {str(d['value']):>18s} {d['unit']:<8s} [{d['tag']}]")
        print(f"                   {d['source']}")
    print()

    report_panel_geometry()
    report_tilt_sensitivity()
    report_instants(H)
    Pm = report_month(H)
    np.save("results/month_mean_power.npy", Pm)
    print("saved results/month_mean_power.npy")