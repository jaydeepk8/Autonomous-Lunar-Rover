import sys
sys.path.insert(0, "src/illumination")

import numpy as np

from sun_geometry import sun_position, SYNODIC_HOURS
from horizon_field import horizon_at, DISC_RADIUS_DEG
from loads import p as load_p, idle_load, total_load, drive_power

FIELD = "results/horizon_field.npy"

PARAMS = {
    "E_batt": dict(
        value=5420.0, unit="Wh", tag="spec",
        source="Bluethmann, NTRS 20240013903: 5420 Wh start of life at 0 C"),
    "dod": dict(
        value=0.80, unit="-", tag="assumed", sweep=(0.60, 1.00),
        source="usable fraction of capacity; no published depth-of-discharge limit"),
    "eta_charge": dict(
        value=0.95, unit="-", tag="assumed", sweep=(0.90, 0.98),
        source="round-trip charge efficiency"),
    "darkness_days": dict(
        value=4.2, unit="days", tag="spec",
        source="VIPER survives complete darkness a little over four Earth days"),
}


def p(name):
    return PARAMS[name]["value"]


def usable_Wh():
    return p("E_batt") * p("dod")


def survival_hours(P_draw):
    return usable_Wh() / P_draw


def implied_hibernate_power():
    return usable_Wh() / (p("darkness_days") * 24.0)


def integrate(P_in, P_out, dt_h, soc0_Wh=None):
    cap = p("E_batt")
    floor = cap * (1.0 - p("dod"))
    soc = cap if soc0_Wh is None else soc0_Wh
    out = np.empty(len(P_in))
    for i in range(len(P_in)):
        net = P_in[i] - P_out[i]
        soc += (net * p("eta_charge") if net > 0 else net) * dt_h
        soc = min(soc, cap)
        out[i] = soc
        if soc <= floor:
            out[i:] = floor
            break
    return out


def solar_series(H, cell, hours):
    r, c = cell
    P = np.empty(len(hours))
    peak = 450.0
    for i, hr in enumerate(hours):
        elev, azim = sun_position(hr)
        h = horizon_at(H, float(azim))[r, c]
        f = np.clip((elev + DISC_RADIUS_DEG - h) / (2.0 * DISC_RADIUS_DEG), 0.0, 1.0)
        P[i] = peak * f * np.cos(np.radians(elev))
    return P


def report_params():
    print("=" * 74)
    print("PARAMETERS")
    print("=" * 74)
    for k, d in PARAMS.items():
        sweep = f"  sweep {d['sweep']}" if "sweep" in d else ""
        print(f"  {k:16s} {d['value']:>9.4g} {d['unit']:<7s} [{d['tag']}]{sweep}")
        print(f"                   {d['source']}")
    print(f"\n  usable capacity: {usable_Wh():.0f} Wh of {p('E_batt'):.0f} Wh")
    print()


def report_survival():
    print("=" * 74)
    print("SURVIVAL TIME IN FULL DARKNESS")
    print("=" * 74)
    print(f"  {'situation':<30s} {'load W':>8s} {'hours':>8s} {'days':>7s}")
    rows = [
        ("hibernating", load_p("P_hibernate")),
        ("parked, shadow heaters", idle_load(shadow=True)),
        ("stalled in fault recovery", idle_load(shadow=True)),
        ("driving flat, no lights", total_load(0.0, True, True, False)),
        ("driving flat, lights on", total_load(0.0, True, True, True)),
    ]
    for name, L in rows:
        h = survival_hours(L)
        print(f"  {name:<30s} {L:8.1f} {h:8.1f} {h / 24:7.2f}")
    print()


def report_published_check():
    print("=" * 74)
    print("CHECK AGAINST THE PUBLISHED DARKNESS FIGURE")
    print("=" * 74)
    P_hib = load_p("P_hibernate")
    mine = survival_hours(P_hib) / 24.0
    pub = p("darkness_days")
    implied = implied_hibernate_power()

    print(f"  assumed hibernate draw   : {P_hib:6.1f} W")
    print(f"  my survival time         : {mine:6.2f} days")
    print(f"  VIPER published          : {pub:6.2f} days")
    print()
    print(f"  implied hibernate draw   : {implied:6.1f} W")
    print(f"  ratio to my assumption   : {implied / P_hib:6.2f} x")
    print()
    if abs(mine - pub) / pub < 0.25:
        print("  WITHIN 25 % -- the load model is consistent with flight hardware")
    else:
        print("  MISMATCH -- the assumed hibernate draw is not consistent with")
        print("  the published survival time. Use the implied value instead:")
        print(f"    P_hibernate = {implied:.0f} W  [derived from spec]")
    print()
    return implied


def report_psr_excursion():
    print("=" * 74)
    print("PSR EXCURSION ENVELOPE  (how long the rover may stay in shadow)")
    print("=" * 74)
    print(f"  {'depth of discharge':>20s} {'no lights':>12s} {'lights on':>12s}")
    L_dark = total_load(0.0, True, True, False)
    L_lit = total_load(0.0, True, True, True)
    for dod in [0.10, 0.20, 0.30, 0.50]:
        E = p("E_batt") * dod
        print(f"  {dod:20.0%} {E / L_dark:11.1f}h {E / L_lit:11.1f}h")
    print()
    print("  A return trip must fit inside these numbers, not a one-way trip.")
    print()


def report_trace(H):
    print("=" * 74)
    print("BATTERY TRACE OVER ONE MONTH")
    print("=" * 74)

    Pm = np.load("results/month_mean_power.npy")
    best = np.unravel_index(np.argmax(Pm), Pm.shape)
    med = np.unravel_index(np.argsort(Pm, axis=None)[Pm.size // 2], Pm.shape)

    hours = np.arange(0.0, SYNODIC_HOURS, 1.0)
    L = idle_load(shadow=True)
    P_out = np.full(len(hours), L)
    floor = p("E_batt") * (1.0 - p("dod"))

    for name, cell in [("best-lit cell", best), ("median cell", med)]:
        P_in = solar_series(H, cell, hours)
        soc = integrate(P_in, P_out, 1.0)
        died = int(np.argmax(soc <= floor)) if (soc <= floor).any() else None
        print(f"  {name} ({int(cell[0])}, {int(cell[1])})")
        print(f"    mean solar in : {P_in.mean():7.1f} W   load {L:.0f} W")
        print(f"    min charge    : {soc.min():7.0f} Wh of {p('E_batt'):.0f}")
        if died is not None:
            print(f"    DIED at hour  : {died} ({died / 24:.1f} days)")
        else:
            print("    survives the full month")
    print()


def survival_map(H, load_W, dt_h=1.0):
    cap = p("E_batt")
    floor = cap * (1.0 - p("dod"))
    eta = p("eta_charge")

    soc = np.full(H.shape[:2], cap, dtype=np.float32)
    alive = np.ones(H.shape[:2], dtype=bool)
    hours_lived = np.zeros(H.shape[:2], dtype=np.float32)

    for hr in np.arange(0.0, SYNODIC_HOURS, dt_h):
        elev, azim = sun_position(hr)
        h = horizon_at(H, float(azim))
        f = np.clip((elev + DISC_RADIUS_DEG - h) / (2.0 * DISC_RADIUS_DEG), 0.0, 1.0)
        net = 450.0 * f * np.cos(np.radians(elev)) - load_W
        soc = np.where(alive, soc + np.where(net > 0, net * eta, net) * dt_h, soc)
        soc = np.minimum(soc, cap)
        hours_lived += alive * dt_h
        alive &= soc > floor

    return alive, hours_lived


def report_survival_map(H):
    print("=" * 74)
    print("WHERE THE ROVER ACTUALLY SURVIVES A MONTH")
    print("=" * 74)
    Pm = np.load("results/month_mean_power.npy")

    print(f"  {'load':>18s} {'mean power OK':>15s} {'really survives':>17s}")
    for name, L in [
        ("hibernating 43 W", implied_hibernate_power()),
        ("parked, heaters", idle_load(shadow=True)),
        ("driving, no lights", total_load(0.0, True, True, False)),
    ]:
        alive, lived = survival_map(H, L)
        print(f"  {name:>18s} {(Pm > L).mean():15.4f} {alive.mean():17.4f}")
    print()
    print("  Left column: cells where month-average sunlight exceeds the load.")
    print("  Right column: cells where the battery actually bridges every gap.")
    print("  The difference is the cost of the month-long shadow cycle.")
    print()

    alive, lived = survival_map(H, idle_load(shadow=True))
    np.save("results/survival_hours.npy", lived)
    print(f"  saved results/survival_hours.npy  "
          f"(median {np.median(lived) / 24:.1f} days)")
    print()


if __name__ == "__main__":
    H = np.load(FIELD)
    print(f"\nhorizon field {H.shape}\n")
    report_params()
    report_survival()
    implied = report_published_check()
    report_psr_excursion()
    report_trace(H)
    report_survival_map(H)