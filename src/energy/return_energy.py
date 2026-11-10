import sys
sys.path.insert(0, "src/illumination")
sys.path.insert(0, "src/energy")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.sparse.csgraph import dijkstra

from sun_geometry import sun_position, SYNODIC_HOURS
from horizon_field import load_field
from irradiance import solar_power
from loads import p as load_p
from edges import terrain, edge_costs, to_graph
from coupling import load_q
from battery import p as batt_p

E_BATT = batt_p("E_batt")
BUDGETS = (0.10, 0.20, 0.30)
KAPPAS = (0.0, 0.5, 1.0)
TIMES = np.arange(0.0, SYNODIC_HOURS, 24.0)
MAP_HOUR = 350.0
CELL_KM2 = 0.04 * 0.04


def fields(terr, H, hour, kappa, q):
    xs, ys = terr["xs"], terr["ys"]
    costs, lights, lit = edge_costs(terr, H, hour, kappa, q)
    G = to_graph(costs)
    G.data = G.data + 1e-9

    elev, azim = sun_position(hour)
    P_sun = solar_power(H, xs, ys, float(elev), float(azim))
    safe = terr["passable"] & (P_sun >= load_p("P_avionics") + load_p("P_heater_sun"))
    src = np.flatnonzero(safe)

    shape = lit.shape
    # cheapest energy from each cell back to the nearest safe cell, and in from one
    e_out = dijkstra(G.T, indices=src, min_only=True).reshape(shape)
    e_in = dijkstra(G, indices=src, min_only=True).reshape(shape)
    return e_in, e_out, lit, safe


def summarise(terr, H, q):
    rows = []
    for hour in TIMES:
        per_k = {k: fields(terr, H, hour, k, q) for k in KAPPAS}
        e_in0, e_out0, lit, safe = per_k[0.0]
        dark = terr["passable"] & ~lit
        if dark.sum() == 0:
            continue
        rt0 = e_in0 + e_out0
        row = dict(hour=hour, safe=safe.mean(), dark=dark.sum(),
                   reachable=np.isfinite(e_out0[dark]).mean())
        for k in KAPPAS:
            e_in, e_out, _, _ = per_k[k]
            rt = e_in + e_out
            ok = dark & np.isfinite(e_out)
            row[("ret", k)] = np.median(e_out[ok]) if ok.any() else np.nan
            for b in BUDGETS:
                row[("area", k, b)] = (dark & (rt <= b * E_BATT)).sum() * CELL_KM2
                if k > 0:
                    row[("lost", k, b)] = (dark & (rt0 <= b * E_BATT) & (rt > b * E_BATT)).sum() * CELL_KM2
        rows.append(row)
    return rows


def med(rows, key):
    v = np.array([r[key] for r in rows], dtype=float)
    return np.nanmedian(v), np.nanmin(v), np.nanmax(v)


def report(rows):
    print("=" * 74)
    print(f"ACROSS THE MONTH  ({len(rows)} moments, one per day; median, then min-max)")
    print("=" * 74)
    m = med(rows, "safe")
    print(f"  site where the rover can sit and charge : {m[0]:.1%}  ({m[1]:.1%}-{m[2]:.1%})")
    m = med(rows, "reachable")
    print(f"  passable shadow it can get back from    : {m[0]:.1%}  ({m[1]:.1%}-{m[2]:.1%})\n")

    print("  energy to get back to a safe cell, median over reachable shadow (Wh)")
    for k in KAPPAS:
        m = med(rows, ("ret", k))
        print(f"    kappa {k:4.2f} : {m[0]:7.0f}  ({m[1]:.0f}-{m[2]:.0f})")
    r0 = np.array([r[("ret", 0.0)] for r in rows])
    r1 = np.array([r[("ret", 1.0)] for r in rows])
    print(f"    kappa 1 needs {np.nanmedian(r1 / r0):.2f}x the kappa 0 reserve\n")

    print("  shadowed ground reachable for a round trip (km2)")
    print(f"    {'budget':>8s}" + "".join(f"  kappa {k:4.2f}" for k in KAPPAS)
          + f"  {'lost at kappa 1':>16s}")
    for b in BUDGETS:
        line = f"    {b:>8.0%}"
        for k in KAPPAS:
            line += f"  {med(rows, ('area', k, b))[0]:10.2f}"
        lost = med(rows, ("lost", 1.0, b))[0]
        base = med(rows, ("area", 0.0, b))[0]
        share = lost / base if base > 0 else np.nan
        line += f"  {lost:7.2f} ({share:5.1%})"
        print(line)
    print()
    print("  'lost at kappa 1': shadow a planner ignoring the coupling would call")
    print("  safe to visit and return from, which is not safe once it is included.\n")


def save_map(terr, H, q):
    e_in0, e_out0, lit, safe = fields(terr, H, MAP_HOUR, 0.0, q)
    e_in1, e_out1, _, _ = fields(terr, H, MAP_HOUR, 1.0, q)
    np.savez("results/return_energy_t350.npz", e_in0=e_in0, e_out0=e_out0,
             e_in1=e_in1, e_out1=e_out1, lit=lit, safe=safe)

    b = 0.30 * E_BATT
    rt0, rt1 = e_in0 + e_out0, e_in1 + e_out1
    cat = np.full(lit.shape, 4, dtype=np.int8)
    cat[rt1 <= b] = 2
    cat[(rt0 <= b) & (rt1 > b)] = 3
    cat[safe] = 1
    cat[~terr["passable"]] = 0

    cols = ["#3a3a3a", "#fcffa4", "#5c2a72", "#f98e09", "#000004"]
    labels = ["too steep", "safe: can sit and charge",
              "round trip fits 30% of battery",
              "fits only if coupling is ignored",
              "out of reach on 30% of battery"]
    fig, ax = plt.subplots(figsize=(8.2, 7.0), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(cat, cmap=ListedColormap(cols),
              norm=BoundaryNorm(np.arange(-0.5, 5.5), len(cols)), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    leg = ax.legend(handles=[Patch(facecolor=c, edgecolor="#666", label=l)
                             for c, l in zip(cols, labels)],
                    loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
                    fontsize=10, labelspacing=1.0)
    for t in leg.get_texts():
        t.set_color("white")
    plt.tight_layout()
    plt.savefig("figures/return_energy.png", dpi=160, facecolor="black",
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved results/return_energy_t350.npz and figures/return_energy.png "
          f"(t = {MAP_HOUR:.0f} h, kappa 0 vs 1, 30% budget)")


if __name__ == "__main__":
    q, _, provenance = load_q()
    H, _, _ = load_field()
    terr = terrain()
    print(f"\nq: {provenance}   fault rate: Lamarre 2023\n")
    rows = summarise(terr, H, q)
    report(rows)
    save_map(terr, H, q)