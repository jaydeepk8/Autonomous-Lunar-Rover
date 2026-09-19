import numpy as np

POWER_MAP = "results/month_mean_power.npy"

PARAMS = {
    "m_rover": dict(
        value=447.0, unit="kg", tag="spec",
        source="Bluethmann, NTRS 20240013903: roving mass 447 kg"),
    "g_moon": dict(
        value=1.62, unit="m/s2", tag="spec",
        source="lunar surface gravity"),
    "v_drive": dict(
        value=0.8 / 3.6, unit="m/s", tag="spec",
        source="VIPER top speed 0.8 km/h"),
    "mu_roll": dict(
        value=0.30, unit="-", tag="assumed", sweep=(0.10, 0.50),
        source="rolling resistance on regolith; Chen et al. soil parameters "
               "cohesion 170 N/m2, friction 35 deg"),
    "eta_drive": dict(
        value=0.70, unit="-", tag="assumed", sweep=(0.50, 0.90),
        source="motor and gearbox efficiency"),
    "P_heater_shadow": dict(
        value=50.0, unit="W", tag="assumed", sweep=(20.0, 200.0),
        source="ACT/NASA JSC: TMS cuts survival heat from hundreds of W to tens"),
    "P_heater_sun": dict(
        value=10.0, unit="W", tag="assumed", sweep=(0.0, 50.0),
        source="reduced heater demand while sunlit"),
    "P_avionics": dict(
        value=50.0, unit="W", tag="assumed", sweep=(30.0, 100.0),
        source="no published figure; Moog IAU and SEPIA wattage not released"),
    "P_light": dict(
        value=40.0, unit="W", tag="assumed", sweep=(10.0, 120.0),
        source="no published figure for the VIPER lighting system"),
    "P_hibernate": dict(
        value=20.0, unit="W", tag="assumed", sweep=(10.0, 50.0),
        source="minimum keep-alive draw"),
}


def p(name):
    return PARAMS[name]["value"]


def drive_power(slope_deg, v=None, mu=None, eta=None, m=None):
    v = p("v_drive") if v is None else v
    mu = p("mu_roll") if mu is None else mu
    eta = p("eta_drive") if eta is None else eta
    m = p("m_rover") if m is None else m
    a = np.radians(slope_deg)
    F = m * p("g_moon") * (np.sin(a) + mu * np.cos(a))
    return np.maximum(F, 0.0) * v / eta


def total_load(slope_deg=0.0, shadow=True, driving=True, lights=False):
    P = p("P_avionics")
    P += p("P_heater_shadow") if shadow else p("P_heater_sun")
    if driving:
        P += drive_power(slope_deg)
    if lights:
        P += p("P_light")
    return P


def idle_load(shadow=True):
    return p("P_avionics") + (p("P_heater_shadow") if shadow else p("P_heater_sun"))


def report_params():
    print("=" * 74)
    print("PARAMETERS")
    print("=" * 74)
    for k, d in PARAMS.items():
        sweep = f"  sweep {d['sweep']}" if "sweep" in d else ""
        print(f"  {k:16s} {d['value']:>9.4g} {d['unit']:<7s} [{d['tag']}]{sweep}")
        print(f"                   {d['source']}")
    print()


def report_drive():
    print("=" * 74)
    print("DRIVE POWER vs SLOPE")
    print("=" * 74)
    lo, hi = PARAMS["mu_roll"]["sweep"]
    print(f"  {'slope':>7s} {'mu=' + str(lo):>10s} {'mu=0.30':>10s} "
          f"{'mu=' + str(hi):>10s}")
    for s in [0, 5, 10, 15, 20, 25]:
        row = [drive_power(s, mu=m) for m in (lo, 0.30, hi)]
        print(f"  {s:7d} {row[0]:10.1f} {row[1]:10.1f} {row[2]:10.1f}")
    print()
    print("  Mechanical drive power only. Commonly quoted '300 W while driving'")
    print("  figures bundle avionics, thermal and margin into one number.")
    print()


def report_budget():
    print("=" * 74)
    print("POWER BUDGET BY SITUATION  (flat ground)")
    print("=" * 74)
    rows = [
        ("sunlit, driving", 0.0, False, True, False),
        ("sunlit, parked", 0.0, False, False, False),
        ("shadow, driving, no lights", 0.0, True, True, False),
        ("shadow, driving, lights on", 0.0, True, True, True),
        ("shadow, stalled in recovery", 0.0, True, False, False),
        ("shadow, hibernating", None, None, None, None),
    ]
    print(f"  {'situation':<30s} {'W':>8s}")
    for name, slope, shadow, driving, lights in rows:
        if slope is None:
            P = p("P_hibernate")
        else:
            P = total_load(slope, shadow, driving, lights)
        print(f"  {name:<30s} {P:8.1f}")
    print()


def report_net(Pm):
    print("=" * 74)
    print("NET POWER  (month-mean solar minus load)")
    print("=" * 74)
    for name, load in [
        ("hibernating", p("P_hibernate")),
        ("parked, sunlit heaters", idle_load(shadow=False)),
        ("parked, shadow heaters", idle_load(shadow=True)),
        ("driving flat, no lights", total_load(0.0, True, True, False)),
        ("driving flat, lights on", total_load(0.0, True, True, True)),
    ]:
        net = Pm - load
        frac = (net > 0).mean()
        print(f"  {name:<26s} load {load:6.1f} W   "
              f"self-sustaining on {frac:.4f} of site")
    print()
    print("  'Self-sustaining' means month-average solar input exceeds that load.")
    print("  It is a necessary condition, not a sufficient one: the battery must")
    print("  also bridge the gaps between lit periods. That is step 2.4.")
    print()


def report_heater_sweep(Pm):
    print("=" * 74)
    print("HEATER SENSITIVITY  (parked in shadow, avionics included)")
    print("=" * 74)
    print(f"  {'P_heater':>9s} {'total load':>11s} {'site self-sustaining':>22s}")
    for h in [20.0, 50.0, 100.0, 150.0, 200.0]:
        load = p("P_avionics") + h
        print(f"  {h:9.0f} {load:11.1f} {(Pm > load).mean():22.4f}")
    print()
    print("  The heater number is unsourced and it moves the answer more than")
    print("  any other parameter. It is the first thing to pin down if a real")
    print("  figure can be found.")
    print()


if __name__ == "__main__":
    Pm = np.load(POWER_MAP)
    print(f"\nmonth-mean power map {Pm.shape}, site mean {Pm.mean():.2f} W\n")
    report_params()
    report_drive()
    report_budget()
    report_net(Pm)
    report_heater_sweep(Pm)