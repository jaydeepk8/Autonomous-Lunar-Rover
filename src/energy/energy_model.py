import sys
sys.path.insert(0, "src/illumination")
sys.path.insert(0, "src/energy")

import json
from contextlib import contextmanager
import numpy as np
from scipy.sparse.csgraph import dijkstra

from sun_geometry import sun_position, SYNODIC_HOURS
from horizon_field import load_field, load_dem, validate
from irradiance import solar_power, PARAMS as SUN_PARAMS
import loads
import edges
import coupling
import battery

E_BATT = battery.PARAMS["E_batt"]["value"]
BUDGET = 0.30
TIMES = np.arange(0.0, SYNODIC_HOURS, 72.0)
CELL_KM2 = 0.04 * 0.04
OUT = "results/energy_model.json"

LIMITATIONS = [
    "Polar summer only: the sun model keeps the subsolar latitude at -1.54 deg. "
    "In polar winter the site gets no sun.",
    "Heater and electronics power have no published source. The shadow heater "
    "(50 W) is higher than the whole hibernation draw (43 W), so a survival "
    "heater and a working heater are not yet separated.",
    "Hibernation power is calculated from VIPER's 4.2-day darkness survival, so "
    "comparing against that figure is not a test. The independent check is the "
    "9.6 h shadow time on 30% of the battery against about 9 h published.",
    "Headlight power has no published source and is swept.",
    "Faults are counted by their average energy cost. A real fault is a whole "
    "recovery stop, so worst cases are worse than these numbers. Phase 3 Monte "
    "Carlo runs are needed for that.",
    "A return trip uses the sunlight at the moment it starts; the sun moves about "
    "0.5 deg per hour while the rover drives.",
    "Each move uses the conditions of the cell it starts from. A cell counts as "
    "sunlit if any part of the sun's disc is visible.",
    "Perception between 105 and 170 deg phase angle is a straight line between "
    "the two measured values; Phase 1 has no data there.",
    "One pooled q is used for all terrain. The per-terrain spread is reported "
    "as a range on the result, not built into the map.",
    "Slopes from a 5 m DEM are noisy, which makes them read steeper. NASA's slope "
    "error map is not used. The rule that a cell is blocked when over half of it "
    "is steeper than 25 deg is assumed.",
    "Panel tilt is modelled for the best case only. Battery capacity loss in the "
    "cold and the +-3.4% yearly change in sunlight strength are not modelled.",
    "The planner grid is 40 m with 8 neighbours, so routes and reach areas carry "
    "grid shape effects.",
    "Rolling resistance and drive efficiency are assumed and swept.",
]


@contextmanager
def setting(table, name, value):
    old = table[name]["value"]
    table[name]["value"] = value
    try:
        yield
    finally:
        table[name]["value"] = old


def headline(terr, H, q, rate="2023"):
    xs, ys = terr["xs"], terr["ys"]
    ratios, shares, lost_km2, area_km2 = [], [], [], []
    for hour in TIMES:
        elev, azim = sun_position(hour)
        P_sun = solar_power(H, xs, ys, float(elev), float(azim))
        safe = terr["passable"] & (P_sun >= loads.p("P_avionics") + loads.p("P_heater_sun"))
        src = np.flatnonzero(safe)
        res = {}
        for kappa in (0.0, 1.0):
            costs, _, lit = edges.edge_costs(terr, H, hour, kappa, q, rate)
            G = edges.to_graph(costs)
            G.data = G.data + 1e-9
            e_out = dijkstra(G.T, indices=src, min_only=True).reshape(lit.shape)
            e_in = dijkstra(G, indices=src, min_only=True).reshape(lit.shape)
            res[kappa] = (e_in + e_out, e_out)
        dark = terr["passable"] & ~lit
        ok = dark & np.isfinite(res[0.0][1])
        if not ok.any():
            continue
        ratios.append(np.median(res[1.0][1][ok]) / np.median(res[0.0][1][ok]))
        b = BUDGET * E_BATT
        a0 = dark & (res[0.0][0] <= b)
        lost = a0 & (res[1.0][0] > b)
        area_km2.append(a0.sum() * CELL_KM2)
        lost_km2.append(lost.sum() * CELL_KM2)
        shares.append(lost.sum() / max(a0.sum(), 1))
    return dict(reserve_ratio=float(np.median(ratios)),
                lost_share=float(np.median(shares)),
                lost_km2=float(np.median(lost_km2)),
                reachable_km2=float(np.median(area_km2)))


def terrain_q(model):
    lo = model["sunlit"]["low_band"]["per_terrain"]
    hi = model["sunlit"]["high_band"]["per_terrain"]
    hd = model["headlight"]["per_terrain"]
    dk = model["dark"]["per_terrain"]
    names = sorted(set(lo) & set(hi) & set(hd) & set(dk))
    return {t: dict(sunlit_low=lo[t], sunlit_high=hi[t], headlight=hd[t], dark=dk[t])
            for t in names}


def run_sweeps(terr, H, q, model):
    rows = [("baseline", "-", headline(terr, H, q))]
    print_row(*rows[0])

    def add(name, value, r):
        rows.append((name, value, r))
        print_row(name, value, r)

    add("fault rate", "Lamarre 2024", headline(terr, H, q, "2024"))
    for name, values in [("P_heater_shadow", (20.0, 100.0, 200.0)),
                         ("P_avionics", (30.0, 100.0)),
                         ("P_light", (10.0, 120.0)),
                         ("mu_roll", (0.10, 0.50))]:
        for v in values:
            with setting(loads.PARAMS, name, v):
                add(name, v, headline(terr, H, q))
    for name, values in [("max_slope", (20.0, 30.0)),
                         ("max_steep_share", (0.25, 0.75))]:
        for v in values:
            with setting(edges.PARAMS, name, v):
                add(name, v, headline(edges.terrain(), H, q))
    if model is not None:
        for t, qt in terrain_q(model).items():
            add("q from terrain", t, headline(terr, H, qt))
    return rows


def print_row(name, value, r):
    print(f"  {name:<18s} {str(value):<22s} {r['reserve_ratio']:8.2f}x "
          f"{r['lost_share']:8.1%} {r['lost_km2']:9.2f} {r['reachable_km2']:10.2f}")


def collect_params():
    out = {}
    for module, table in [("loads", loads.PARAMS), ("edges", edges.PARAMS),
                          ("battery", battery.PARAMS), ("coupling", coupling.PARAMS),
                          ("irradiance", SUN_PARAMS)]:
        for k, d in table.items():
            out[k] = {kk: (list(vv) if isinstance(vv, tuple) else vv)
                      for kk, vv in d.items()}
            out[k]["file"] = f"src/energy/{module}.py"
    return out


if __name__ == "__main__":
    q, model, provenance = coupling.load_q()
    H, xs, ys = load_field()
    terr = edges.terrain()
    *_, tb, _, _ = load_dem()

    print("\nSUNLIGHT MODEL")
    iou = validate(H, xs, ys, tb)

    Pm = np.load("results/month_mean_power.npy")
    print(f"\nHEADLINE SWEEP  (budget {BUDGET:.0%} of battery, {len(TIMES)} moments, "
          f"kappa 0 vs 1, one change at a time)")
    print(f"  {'changed':<18s} {'to':<22s} {'reserve':>9s} {'lost':>8s} "
          f"{'lost km2':>9s} {'reach km2':>10s}")
    rows = run_sweeps(terr, H, q, model)

    base = rows[0][2]
    rr = [r["reserve_ratio"] for _, _, r in rows]
    ls = [r["lost_share"] for _, _, r in rows]
    print(f"\n  reserve at kappa 1 vs kappa 0 : {base['reserve_ratio']:.2f}x "
          f"(range {min(rr):.2f}-{max(rr):.2f}x)")
    print(f"  shadow wrongly called safe    : {base['lost_share']:.1%} "
          f"(range {min(ls):.1%}-{max(ls):.1%})")

    out = {
        "what": "energy model for a VIPER-class rover at PGDA Site04, polar summer",
        "grid": {"cell_m": float(xs[1] - xs[0]), "shape": list(H.shape[:2]),
                 "neighbours": 8},
        "parameters": collect_params(),
        "checks": {
            "psr_iou_vs_nasa": round(float(iou), 4),
            "hibernation_W_derived_from_4_2_day_survival":
                round(battery.implied_hibernate_power(), 1),
            "shadow_hours_on_30pct_battery":
                round(0.30 * E_BATT / loads.total_load(0.0, True, True, False), 1),
            "shadow_hours_published_viper": 9.0,
        },
        "sunlight": {
            "site_mean_power_W": round(float(Pm.mean()), 2),
            "never_lit_fraction": round(float((Pm == 0).mean()), 4),
        },
        "headline": {
            "budget_fraction_of_battery": BUDGET,
            "fault_rate": "Lamarre 2023 unless changed",
            "baseline": base,
            "sweep": [{"changed": n, "to": v, **r} for n, v, r in rows[1:]],
            "reserve_ratio_range": [min(rr), max(rr)],
            "lost_share_range": [min(ls), max(ls)],
        },
        "limitations": LIMITATIONS,
        "q_provenance": provenance,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")