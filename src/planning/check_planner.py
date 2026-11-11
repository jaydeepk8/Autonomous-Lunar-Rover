# checker or validator of planner.py

import sys
sys.path.insert(0, "src/illumination")
sys.path.insert(0, "src/energy")
sys.path.insert(0, "src/planning")

import json
import os
import time
import numpy as np

from sun_geometry import SYNODIC_HOURS
from horizon_field import load_field
from edges import terrain, edge_costs, STEPS
from coupling import load_q, PARAMS as FAULT
from loads import p as load_p
from planner import (Planner, PARAMS, budget_Wh, charge_power, outbound_path,
                     home_path, path_cost, cell)

TIMES = np.arange(0.0, SYNODIC_HOURS, 72.0)   # same moments as energy_model.py
RATES = ("2023", "2024")
KAPPA_AWARE = 1.0
CELL_KM2 = 0.04 * 0.04
EXAMPLE_HOUR = 350.0
PHASE2 = "results/energy_model.json"
EXAMPLE_OUT = "results/planner_example.npz"


def report_params():
    print("=" * 74)
    print("PARAMETERS")
    print("=" * 74)
    for k, d in PARAMS.items():
        sweep = f"  sweep {d['sweep']}" if "sweep" in d else ""
        print(f"  {k:12s} {d['value']:>8.4g} {d['unit']:<4s} [{d['tag']}]{sweep}")
        print(f"               {d['source']}")
    print(f"\n  sortie budget   : {budget_Wh():.0f} Wh")
    print(f"  charging cell   : sunlight >= {charge_power():.0f} W "
          f"(avionics + sunlit heater)\n")


def report_lights(terr, H, q):
    print("=" * 74)
    print(f"HEADLIGHT RULE  (lights on in shadow when break-even > {load_p('P_light'):.0f} W)")
    print("=" * 74)
    print(f"  {'kappa':>6s}" + "".join(f"  {'Lamarre ' + r:>18s}" for r in RATES))
    for k in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
        line = f"  {k:6.2f}"
        for r in RATES:
            pl = Planner("x", terr, H, q, k, r)
            line += f"  {pl.P_be:7.1f} W {'ON ' if pl.lights else 'off':>4s}    "
        print(line)
    print("\n  the blind planner (kappa 0) never turns lights on\n")


def departures(terr, H, q, rate):
    blind = Planner("blind", terr, H, q, 0.0, rate)
    aware = Planner("aware", terr, H, q, KAPPA_AWARE, rate)
    passable = terr["passable"].ravel()
    b = budget_Wh()
    rows, secs = [], []
    for hour in TIMES:
        t0 = time.time()
        d0 = blind.departure(hour)
        d1 = aware.departure(hour)
        secs.append((time.time() - t0) / 2.0)

        dark = passable & ~d0["lit"].ravel()
        ok = dark & np.isfinite(d0["home"])
        if not ok.any():
            continue
        rt0, rt1 = blind.round_trip(d0), aware.round_trip(d1)
        a0 = dark & (rt0 <= b)
        lost = a0 & (rt1 > b)
        rows.append(dict(
            hour=hour, azim=d0["azim"], n_charge=d0["n_charge"],
            dark=int(dark.sum()), reach=ok.sum() / dark.sum(),
            ratio=np.median(d1["home"][ok]) / np.median(d0["home"][ok]),
            a0_km2=a0.sum() * CELL_KM2, a1_km2=(dark & (rt1 <= b)).sum() * CELL_KM2,
            lost_km2=lost.sum() * CELL_KM2, share=lost.sum() / max(a0.sum(), 1)))
    return blind, aware, rows, secs


def report_departures(rate, rows):
    print("=" * 74)
    print(f"DEPARTURE MOMENTS  (Lamarre {rate}, blind kappa 0 vs aware kappa {KAPPA_AWARE:g}, "
          f"budget {budget_Wh():.0f} Wh)")
    print("=" * 74)
    print(f"  {'hour':>5s} {'azim':>6s} {'charge km2':>11s} {'shadow km2':>11s} "
          f"{'home ok':>8s} {'blind km2':>10s} {'aware km2':>10s} {'rejected':>9s}")
    for r in rows:
        print(f"  {r['hour']:5.0f} {r['azim']:6.1f} {r['n_charge'] * CELL_KM2:11.2f} "
              f"{r['dark'] * CELL_KM2:11.2f} {r['reach']:8.1%} {r['a0_km2']:10.2f} "
              f"{r['a1_km2']:10.2f} {r['share']:9.1%}")
    print("\n  home ok  : share of passable shadow with any route to a charging cell")
    print("  rejected : share of blind-accepted shadow the aware planner turns down\n")


def phase2_reference():
    if not os.path.exists(PHASE2):
        return None
    with open(PHASE2) as f:
        h = json.load(f)["headline"]
    ref = {"2023": h["baseline"]}
    for s in h["sweep"]:
        if s["changed"] == "fault rate":
            ref["2024"] = s
    return ref


def report_link(results, ref):
    print("=" * 74)
    print("LINK TO PHASE 2  (median over the departure moments)")
    print("=" * 74)
    ok_all = True
    for rate, rows in results.items():
        ratio = float(np.median([r["ratio"] for r in rows]))
        share = float(np.median([r["share"] for r in rows]))
        lost = float(np.median([r["lost_km2"] for r in rows]))
        reach = float(np.median([r["a0_km2"] for r in rows]))
        print(f"  Lamarre {rate}")
        print(f"    reserve ratio   : {ratio:.4f}x")
        print(f"    rejected share  : {share:.4f}")
        print(f"    lost / reach    : {lost:.2f} / {reach:.2f} km2")
        if ref is None or rate not in ref:
            print("    Phase 2 value   : not found, cannot compare\n")
            ok_all = False
            continue
        e = ref[rate]
        good = abs(ratio - e["reserve_ratio"]) < 0.005 and abs(share - e["lost_share"]) < 0.002
        ok_all &= good
        print(f"    Phase 2 value   : {e['reserve_ratio']:.4f}x, {e['lost_share']:.4f}")
        print(f"    {'MATCH' if good else 'MISMATCH -- stop here, the planner disagrees with Phase 2'}\n")
    return ok_all


def report_routes(terr, H, q, blind, aware, rate, n=200, seed=0):
    print("=" * 74)
    print(f"ROUTES MATCH THEIR COSTS  (Lamarre {rate}, t = {TIMES[0]:.0f} h, {n} random targets)")
    print("=" * 74)
    shape = terr["passable"].shape
    rng = np.random.default_rng(seed)
    ok_all = True
    for pl in (blind, aware):
        dep = pl.departure(TIMES[0])
        costs = edge_costs(terr, H, TIMES[0], pl.kappa, q, rate)[0]
        dark = terr["passable"].ravel() & ~dep["lit"].ravel()
        cand = np.flatnonzero(dark & np.isfinite(dep["out"]) & np.isfinite(dep["home"]))
        if cand.size == 0:
            print(f"  {pl.name}: no reachable shadow at this moment\n")
            continue
        pick = rng.choice(cand, size=min(n, cand.size), replace=False)
        err, ends_ok = 0.0, True
        for t in pick:
            out, home = outbound_path(dep, t), home_path(dep, t)
            ends_ok &= bool(dep["charge"].ravel()[out[0]] and dep["charge"].ravel()[home[-1]])
            err = max(err, abs(path_cost(out, costs, shape) - dep["out"][t]) / dep["out"][t],
                      abs(path_cost(home, costs, shape) - dep["home"][t]) / dep["home"][t])
        good = err < 1e-6 and ends_ok
        ok_all &= good
        print(f"  {pl.name:<6s} kappa {pl.kappa:3.1f}: worst cost mismatch {err:.1e}, "
              f"routes start and end on charging cells: {ends_ok}  "
              f"{'PASS' if good else 'FAIL'}")

    same = Planner("aware0", terr, H, q, 0.0, rate).departure(TIMES[0])
    d0 = blind.departure(TIMES[0])
    ident = all(np.array_equal(same[k], d0[k]) for k in ("out", "home", "out_pred", "home_pred"))
    ok_all &= ident
    print(f"  aware planner at kappa 0 identical to blind: {ident}  {'PASS' if ident else 'FAIL'}\n")
    return ok_all


def route_km(path, terr):
    shape = terr["passable"].shape
    L = 0.0
    for a, b in zip(path[:-1], path[1:]):
        r, c = cell(a, shape)
        (r1, c1) = cell(b, shape)
        k = STEPS.index((r1 - r, c1 - c))
        L += terr["steps"][k]["L"][r, c]
    return L / 1000.0


def report_example(terr, H, q, rate="2023"):
    print("=" * 74)
    print(f"EXAMPLE TARGET  (Lamarre {rate}, t = {EXAMPLE_HOUR:.0f} h)")
    print("=" * 74)
    shape = terr["passable"].shape
    blind = Planner("blind", terr, H, q, 0.0, rate)
    aware = Planner("aware", terr, H, q, KAPPA_AWARE, rate)
    d0, d1 = blind.departure(EXAMPLE_HOUR), aware.departure(EXAMPLE_HOUR)
    lit = d0["lit"].ravel()
    dark = terr["passable"].ravel() & ~lit
    rt0, rt1 = blind.round_trip(d0), aware.round_trip(d1)
    lost = np.flatnonzero(dark & (rt0 <= budget_Wh()) & (rt1 > budget_Wh()))
    if lost.size == 0:
        print("  no target here that only the blind planner accepts\n")
        return
    t = lost[np.argsort(rt0[lost])[lost.size // 2]]

    print(f"  target cell {cell(t, shape)}, a cell the blind planner accepts and the aware "
          f"one rejects\n")
    print(f"  {'':8s} {'out km':>7s} {'home km':>8s} {'out Wh':>8s} {'home Wh':>8s} "
          f"{'total Wh':>9s} {'in shadow':>10s} {'lights':>7s} {'verdict':>8s}")
    paths = {}
    for pl, d in ((blind, d0), (aware, d1)):
        out, home = outbound_path(d, t), home_path(d, t)
        paths[pl.name] = (out, home)
        both = out + home[1:]
        shade = 1.0 - lit[both].mean()
        total = d["out"][t] + d["home"][t]
        print(f"  {pl.name:<8s} {route_km(out, terr):7.2f} {route_km(home, terr):8.2f} "
              f"{d['out'][t]:8.0f} {d['home'][t]:8.0f} {total:9.0f} {shade:10.0%} "
              f"{'on' if pl.lights else 'off':>7s} "
              f"{'accept' if total <= budget_Wh() else 'reject':>8s}")
    print(f"\n  budget {budget_Wh():.0f} Wh; 'in shadow' is the share of route cells dark "
          f"at departure\n")

    np.savez(EXAMPLE_OUT, hour=EXAMPLE_HOUR, target=t, shape=shape,
             lit=d0["lit"], charge=d0["charge"], passable=terr["passable"],
             blind_out=paths["blind"][0], blind_home=paths["blind"][1],
             aware_out=paths["aware"][0], aware_home=paths["aware"][1])
    print(f"  saved {EXAMPLE_OUT}\n")


def report_timing(terr, H, q, secs):
    print("=" * 74)
    print("TIMING")
    print("=" * 74)
    pl = Planner("t", terr, H, q, 0.0)
    t0 = time.time()
    pl.view(1.0)
    one = time.time() - t0
    print(f"  departure build (two searches): {np.mean(secs):.2f} s each")
    print(f"  replanning view (one search)  : {one:.2f} s each\n")


def run(terr, H, q, ref):
    report_params()
    report_lights(terr, H, q)
    results, checks, secs = {}, [], []
    for rate in RATES:
        blind, aware, rows, s = departures(terr, H, q, rate)
        results[rate] = rows
        secs += s
        report_departures(rate, rows)
        checks.append(("routes match costs, " + rate,
                       report_routes(terr, H, q, blind, aware, rate)))
    checks.insert(0, ("matches Phase 2", report_link(results, ref)))
    report_example(terr, H, q)
    report_timing(terr, H, q, secs)

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for name, good in checks:
        print(f"  {name:<28s} {'PASS' if good else 'FAIL'}")
    print()
    return results


if __name__ == "__main__":
    q, _, provenance = load_q()
    H, _, _ = load_field()
    terr = terrain()
    print(f"\nq: {provenance}\ngrid {terr['passable'].shape}, "
          f"passable {terr['passable'].mean():.1%}\n")
    run(terr, H, q, phase2_reference())