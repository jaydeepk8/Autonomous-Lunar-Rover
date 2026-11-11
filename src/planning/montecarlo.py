import sys
sys.path.insert(0, "src/illumination")
sys.path.insert(0, "src/energy")
sys.path.insert(0, "src/planning")

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import numpy as np
import pandas as pd

import sun_geometry
from sun_geometry import SYNODIC_HOURS
from horizon_field import load_field
from edges import terrain, PARAMS as EDGE_PARAMS
from coupling import load_q, PARAMS as FAULT
from planner import Planner, PARAMS as PLAN_PARAMS, budget_Wh
from execute import World, fly, settings, PARAMS as EXEC_PARAMS
from loads import PARAMS as LOAD_PARAMS
from battery import usable_Wh

TIMES = np.arange(0.0, SYNODIC_HOURS, 72.0)   # same moments as Phase 2
RATES = ("2023", "2024")
SEED = 11
OUTCOMES = ("within budget", "turned back", "over budget", "stranded")
P_HIB_50H = round(usable_Wh() / 50.0, 1)   # hibernation draw NASA's 50 h darkness limit implies

CONFIG = {
    "full": dict(n_targets=60, n_runs=20, times=TIMES,
                 runs="results/mission_runs.csv", done="results/mission_done.json",
                 out="results/mission_model.json"),
    "quick": dict(n_targets=8, n_runs=4, times=TIMES[::4],
                  runs="results/mission_runs_quick.csv", done="results/mission_done_quick.json",
                  out="results/mission_model_quick.json"),
}

LIMITATIONS = [
    "Baseline is peak south polar summer (subsolar latitude -1.54 deg); the shoulder "
    "season (-1.0 and -0.5 deg) is run at kappa 0 and 1 only.",
    "Baseline average speed equals VIPER's driving speed (22 cm/s). VIPER is planned at "
    "about 1 cm/s speed made good; 5 and 2 cm/s are run at kappa 0 and 1 only. Stopped "
    "time is charged at idle load with lights off.",
    "The planner holds the sun fixed for 2 h between refreshes, and refreshes at once "
    "when the rover finds its charging cell has lost the sun.",
    "The turn-back check uses the rest of the departure plan for the outbound leg and "
    "the current plan for the way home.",
    "Faults are served at the end of the move where they occur, at 100 W idle with lights "
    "off, for a fixed 5 h (2023) or 10 h (2024). Recovery never fails.",
    "Sorties start and end on any cell lit at that moment. Whether it stays lit (NASA's "
    "safe havens must have no darkness longer than 50 h) is not checked, so some rovers "
    "are caught when their whole reachable area goes dark; these are reported separately "
    "as stranded while waiting for the sun.",
    "When no charging cell can be reached, the rover hibernates where it is and waits for "
    "the sun. Hibernation is 43 W (from VIPER's 4.2-day darkness survival); NASA's 50 h "
    "darkness limit implies about 87 W, which is run at kappa 0 and 1 only. The two "
    "figures are not yet reconciled.",
    "The coupling-aware planner knows the true kappa exactly; this is its best case.",
    "Each sortie leaves on a full battery with a 30% budget and no parking time at the "
    "target. Stranding means the battery reaches the 20% floor, or the rover is out over 240 h.",
    "Targets are drawn evenly from shadow the blind planner accepts, at 10 moments of one "
    "month, at one site.",
    "Carried from Phase 2: 40 m grid with 8 neighbours, one pooled q, q between 105 and "
    "170 deg phase interpolated, heater, electronics and headlight power assumed.",
]


def scenarios():
    out = []
    base = dict(v_eff=None, season=None, p_hib=None)
    for rate in RATES:
        out.append(dict(base, name="main", rate=rate, kappas=(0.0, 0.5, 1.0, 2.0)))
    for rate in RATES:
        for v in EDGE_PARAMS["v_eff"]["sweep"]:
            out.append(dict(base, name="speed", rate=rate, kappas=(0.0, 1.0), v_eff=v))
    for rate in RATES:
        for s in (-1.0, -0.5):
            out.append(dict(base, name="season", rate=rate, kappas=(0.0, 1.0), season=s))
    for rate in RATES:
        out.append(dict(base, name="hibernate", rate=rate, kappas=(0.0, 1.0), p_hib=P_HIB_50H))
    return out


def row(sc, speed, season, p_hib, hour, kappa, planner, target, run, r, accepts, rt_b, rt_a):
    return dict(scenario=sc["name"], rate=sc["rate"], speed_cm_s=round(speed * 100, 2),
                season_deg=season, hibernate_W=p_hib, kappa=kappa, planner=planner, hour=hour,
                target=int(target),
                run=run, aware_accepts=bool(accepts), rt_blind=float(rt_b), rt_aware=float(rt_a),
                predicted=r["predicted"], outcome=r["outcome"], reason=r["reason"] or "",
                reached=r["reached"], turned_back=r["turned_back"], breach=r["breach"],
                drawdown=r["drawdown"], faults=r["faults"], expected_faults=r["expected_faults"],
                hours=r["hours"], reroutes=r["reroutes"], waits=r["waits"])


def identity_check(terr, H, q, cfg):
    # an aware planner at kappa 0 must behave exactly like the blind one
    pb, pa = Planner("blind", terr, H, q, 0.0), Planner("aware0", terr, H, q, 0.0)
    world = World(terr, H, q)
    hour = float(cfg["times"][0])
    db, da = pb.departure(hour), pa.departure(hour)
    dark = terr["passable"].ravel() & ~db["lit"].ravel()
    cand = np.flatnonzero(dark & pb.accepts(db))[:10]
    same = True
    for t in cand:
        for run in range(2):
            a = fly(world, pb, db, int(t), 0.0, rng=np.random.default_rng([SEED, 0, int(t), run]))
            b = fly(world, pa, da, int(t), 0.0, rng=np.random.default_rng([SEED, 0, int(t), run]))
            same &= a["outcome"] == b["outcome"] and a["drawdown"] == b["drawdown"]
    print(f"  aware planner at kappa 0 behaves exactly like the blind one: {same}\n")
    return bool(same)


_W = {}


def load_inputs():
    q, _, provenance = load_q()
    H, _, _ = load_field()
    return dict(q=q, H=H, terr=terrain(), provenance=provenance)


def init_worker():
    # on Windows each worker starts fresh and loads the inputs once
    if not _W:
        _W.update(load_inputs())


def block_values(sc):
    with settings(v_eff=sc["v_eff"], season=sc["season"], p_hibernate=sc["p_hib"]):
        return (EDGE_PARAMS["v_eff"]["value"], sun_geometry.SEASON["subsolar_lat_deg"],
                LOAD_PARAMS["P_hibernate"]["value"])


def block_key(sc, hour):
    speed, season, p_hib = block_values(sc)
    return f"{sc['name']}|{sc['rate']}|{speed:.4f}|{season:.2f}|{p_hib:.1f}|{hour:.0f}"


def run_block(task):
    """One scenario at one departure moment: pick targets, fly every run for every kappa."""
    sc, di, hour, n_targets, n_runs = task
    terr, H, q = _W["terr"], _W["H"], _W["q"]
    passable = terr["passable"].ravel()
    t0 = time.time()
    rows, n_tg = [], 0
    with settings(v_eff=sc["v_eff"], season=sc["season"], p_hibernate=sc["p_hib"]):
        speed = EDGE_PARAMS["v_eff"]["value"]
        season = sun_geometry.SEASON["subsolar_lat_deg"]
        p_hib = LOAD_PARAMS["P_hibernate"]["value"]
        world = World(terr, H, q, sc["rate"])
        blind = Planner("blind", terr, H, q, 0.0, sc["rate"])
        aware = {k: Planner("aware", terr, H, q, k, sc["rate"]) for k in sc["kappas"] if k > 0}

        d0 = blind.departure(hour)
        dark = passable & ~d0["lit"].ravel()
        cand = np.flatnonzero(dark & blind.accepts(d0))
        if cand.size:
            pick = np.random.default_rng([SEED, di])
            targets = pick.choice(cand, size=min(n_targets, cand.size), replace=False)
            n_tg = len(targets)
            rt0 = blind.round_trip(d0)
            for k in sc["kappas"]:
                pa = aware.get(k)
                da = pa.departure(hour) if pa else d0
                acc = pa.accepts(da) if pa else np.ones(passable.size, dtype=bool)
                rta = pa.round_trip(da) if pa else rt0
                for t in targets:
                    for run in range(n_runs):
                        seed = [SEED, di, int(t), run]
                        rb = fly(world, blind, d0, int(t), k, rng=np.random.default_rng(seed))
                        args = (sc, speed, season, p_hib, hour, k)
                        rows.append(row(*args, "blind", t, run, rb, acc[t], rt0[t], rta[t]))
                        if pa is None:
                            rows.append(row(*args, "aware", t, run, rb, True, rt0[t], rt0[t]))
                        elif acc[t]:
                            ra = fly(world, pa, da, int(t), k, rng=np.random.default_rng(seed))
                            rows.append(row(*args, "aware", t, run, ra, True, rt0[t], rta[t]))
    label = (f"{sc['name']:<9s} {sc['rate']}  {speed * 100:5.1f} cm/s  season {season:5.2f}  "
             f"hibernate {p_hib:4.1f} W  t = {hour:3.0f} h  targets {n_tg:3d}  sorties {len(rows):6d}")
    return block_key(sc, hour), rows, label, time.time() - t0


def run_all(tasks, cfg, done, workers):
    total, t_start = len(tasks), time.time()
    if workers > 1:
        pool = mp.Pool(workers, initializer=init_worker)
        results = pool.imap_unordered(run_block, tasks)
    else:
        pool, results = None, map(run_block, tasks)
    try:
        for i, (key, rows, label, secs) in enumerate(results, 1):
            if rows:
                pd.DataFrame(rows).to_csv(cfg["runs"], mode="a",
                                          header=not os.path.exists(cfg["runs"]), index=False)
            done.add(key)
            with open(cfg["done"], "w") as f:
                json.dump(sorted(done), f)
            el = time.time() - t_start
            eta = el / i * (total - i)
            print(f"  [{i:3d}/{total}] {label}  {secs:6.1f} s   elapsed {el / 60:5.1f} min, "
                  f"left ~{eta / 60:5.1f} min", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(c - h, 0.0), min(c + h, 1.0))


def summarise(df):
    groups = []
    keys = ["scenario", "rate", "speed_cm_s", "season_deg", "hibernate_W", "kappa"]
    for gk, g in df.groupby(keys, sort=True):
        b = g[g.planner == "blind"]
        offered = b[["hour", "target"]].drop_duplicates().shape[0]
        for planner in ("blind", "aware"):
            s = g[g.planner == planner]
            n = len(s)
            if n == 0:
                continue
            flown = s[["hour", "target"]].drop_duplicates().shape[0]
            acc = flown / offered if offered else math.nan
            dd = s.drawdown / budget_Wh()
            e = s.expected_faults.sum()
            out = dict(zip(keys, gk))
            out.update(planner=planner, targets_offered=int(offered), accepted=acc, sorties=int(n),
                       drawdown_vs_budget=dict(p50=float(dd.quantile(0.5)),
                                               p90=float(dd.quantile(0.9)),
                                               p99=float(dd.quantile(0.99))),
                       faults_mean=float(s.faults.mean()), faults_expected=float(s.expected_faults.mean()),
                       faults_z=float((s.faults.sum() - e) / math.sqrt(e)) if e > 0 else 0.0,
                       hours_mean=float(s.hours.mean()),
                       stranded_after_waiting=int(((s.outcome == "stranded") & (s.waits > 0)).sum()),
                       share={}, ci95={}, per_100_offered={})
            for o in OUTCOMES:
                k = int((s.outcome == o).sum())
                out["share"][o] = k / n
                out["ci95"][o] = list(wilson(k, n))
                out["per_100_offered"][o] = 100.0 * acc * k / n
            out["per_100_offered"]["declined"] = 100.0 * (1.0 - acc)
            if planner == "blind" and gk[-1] > 0:
                split = {}
                for name, m in (("aware_declines", ~s.aware_accepts), ("aware_accepts", s.aware_accepts)):
                    ss = s[m]
                    split[name] = dict(sorties=int(len(ss)),
                                       **{o: float((ss.outcome == o).mean()) if len(ss) else math.nan
                                          for o in OUTCOMES})
                out["split"] = split
            groups.append(out)
    return groups


def print_report(groups):
    head = (f"  {'kappa':>5s} {'planner':>7s} {'accept':>7s} {'sorties':>8s} {'within':>7s} "
            f"{'turned':>7s} {'over':>13s} {'stranded':>13s} {'waited':>6s} {'p50':>5s} {'p90':>5s} "
            f"{'p99':>5s} {'faults':>6s}")
    last = None
    for gr in groups:
        sec = (gr["scenario"], gr["rate"], gr["speed_cm_s"], gr["season_deg"], gr["hibernate_W"])
        if sec != last:
            print("=" * 100)
            print(f"{gr['scenario'].upper()}  Lamarre {gr['rate']}, speed {gr['speed_cm_s']} cm/s, "
                  f"season {gr['season_deg']} deg, hibernate {gr['hibernate_W']} W  "
                  f"(energy used vs budget: p50 p90 p99)")
            print("=" * 100)
            print(head)
            last = sec
        sh, ci = gr["share"], gr["ci95"]
        dd = gr["drawdown_vs_budget"]
        print(f"  {gr['kappa']:5.1f} {gr['planner']:>7s} {gr['accepted']:7.1%} {gr['sorties']:8d} "
              f"{sh['within budget']:7.1%} {sh['turned back']:7.1%} "
              f"{sh['over budget']:5.1%} ({ci['over budget'][0]:4.1%}-{ci['over budget'][1]:4.1%}) "
              f"{sh['stranded']:5.1%} ({ci['stranded'][0]:4.1%}-{ci['stranded'][1]:4.1%}) "
              f"{gr['stranded_after_waiting'] / gr['sorties']:6.1%} "
              f"{dd['p50']:5.2f} {dd['p90']:5.2f} {dd['p99']:5.2f} {gr['faults_mean']:6.2f}")
    print("  waited: stranded after hibernating because no lit charging cell could be reached")
    print("  (the base went dark); counted inside 'stranded' with the fault-driven cases\n")


def scenario_label(gr):
    return {"main": "main",
            "speed": f"speed {gr['speed_cm_s']:g} cm/s",
            "season": f"season {gr['season_deg']:g}",
            "hibernate": f"hibernate {gr['hibernate_W']:g} W"}[gr["scenario"]]


def print_per_100(groups):
    print("=" * 100)
    print("PER 100 TARGETS OFFERED  (main scenario)")
    print("=" * 100)
    print(f"  {'rate':>5s} {'kappa':>5s} {'planner':>7s} {'declined':>9s} {'within':>7s} "
          f"{'turned':>7s} {'over':>6s} {'stranded':>9s}")
    for gr in groups:
        if gr["scenario"] != "main":
            continue
        p = gr["per_100_offered"]
        print(f"  {gr['rate']:>5s} {gr['kappa']:5.1f} {gr['planner']:>7s} {p['declined']:9.1f} "
              f"{p['within budget']:7.1f} {p['turned back']:7.1f} {p['over budget']:6.1f} "
              f"{p['stranded']:9.1f}")
    print()
    print("=" * 100)
    print("BLIND PLANNER, SPLIT BY WHETHER THE AWARE PLANNER WOULD HAVE FLOWN THE TARGET")
    print("=" * 100)
    print(f"  {'scenario':>16s} {'rate':>5s} {'kappa':>5s} {'targets':>16s} {'sorties':>8s} "
          f"{'within':>7s} {'turned':>7s} {'over':>6s} {'stranded':>9s}")
    for gr in groups:
        if "split" not in gr:
            continue
        for name, s in gr["split"].items():
            label = "aware declines" if name == "aware_declines" else "aware accepts"
            if s["sorties"] == 0:
                continue
            print(f"  {scenario_label(gr):>16s} {gr['rate']:>5s} {gr['kappa']:5.1f} {label:>16s} "
                  f"{s['sorties']:8d} {s['within budget']:7.1%} {s['turned back']:7.1%} "
                  f"{s['over budget']:6.1%} {s['stranded']:9.1%}")
    print()


def headline(groups):
    out = {}
    for rate in RATES:
        g = {(x["kappa"], x["planner"]): x for x in groups
             if x["scenario"] == "main" and x["rate"] == rate}
        if (0.0, "blind") not in g or (1.0, "blind") not in g:
            continue
        b0, b1, a1 = g[(0.0, "blind")], g[(1.0, "blind")], g.get((1.0, "aware"))
        def pick(x):
            return dict(over=x["share"]["over budget"], stranded=x["share"]["stranded"],
                        stranded_waiting=x["stranded_after_waiting"] / x["sorties"])
        out[rate] = dict(blind_kappa0=pick(b0), blind_kappa1=pick(b1),
                         aware_kappa1=None if a1 is None else dict(pick(a1), declined=1.0 - a1["accepted"]))
    return out


def print_headline(hl):
    print("=" * 100)
    print("HEADLINE  (main scenario: peak summer, 22 cm/s)")
    print("=" * 100)
    for rate, h in hl.items():
        b0, b1, a1 = h["blind_kappa0"], h["blind_kappa1"], h["aware_kappa1"]
        print(f"  Lamarre {rate}")
        for label, x in (("blind, model right (kappa 0)  ", b0), ("blind, coupling real (kappa 1)", b1),
                         ("aware (kappa 1)               ", a1)):
            if x is None:
                continue
            extra = f", declines {x['declined']:5.1%} of targets" if "declined" in x else ""
            print(f"    {label}: over budget {x['over']:5.1%}, stranded {x['stranded']:5.1%} "
                  f"({x['stranded_waiting']:4.1%} waiting for sun){extra}")
    print()


def checks(groups, ident):
    worst = max(abs(g["faults_z"]) for g in groups)
    k0 = [g for g in groups if g["kappa"] == 0.0]
    pairs = {}
    for g in k0:
        pairs.setdefault((g["scenario"], g["rate"], g["speed_cm_s"], g["season_deg"],
                          g["hibernate_W"]), {})[g["planner"]] = g
    k0_same = all(p["blind"]["share"] == p["aware"]["share"] for p in pairs.values() if len(p) == 2)
    res = {"fault counts match rate (|z| < 4 in every group)": bool(worst < 4.0),
           "kappa 0: aware planner identical to blind": bool(ident and k0_same)}
    print("=" * 100)
    print("CHECKS")
    print("=" * 100)
    print(f"  worst fault-count z over {len(groups)} groups: {worst:.2f}")
    n_st = sum(round(g["share"]["stranded"] * g["sorties"]) for g in groups)
    n_wait = sum(g["stranded_after_waiting"] for g in groups)
    print(f"  stranded sorties that first waited for the sun: {n_wait} of {n_st}")
    for k, v in res.items():
        print(f"  {k:<52s} {'PASS' if v else 'FAIL'}")
    print()
    return res


def collect_params(cfg):
    out = {}
    for module, table in (("planner", PLAN_PARAMS), ("execute", EXEC_PARAMS)):
        for k, d in table.items():
            out[k] = {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in d.items()}
            out[k]["file"] = f"src/planning/{module}.py"
    v = EDGE_PARAMS["v_eff"]
    out["v_eff"] = {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()}
    out["v_eff"]["file"] = "src/energy/edges.py"
    out["season"] = dict(value=sun_geometry.SEASON["subsolar_lat_deg"], unit="deg", tag="assumed",
                         sweep=[-1.0, -0.5], source="subsolar latitude; -1.54 is peak south polar summer",
                         file="src/illumination/sun_geometry.py")
    hib = LOAD_PARAMS["P_hibernate"]
    out["P_hibernate"] = {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in hib.items()}
    out["P_hibernate"].update(sweep=[P_HIB_50H], file="src/energy/loads.py",
                              use="draw while hibernating to wait for the sun")
    out["monte_carlo"] = dict(departures=[float(t) for t in cfg["times"]], n_targets=cfg["n_targets"],
                              n_runs=cfg["n_runs"], seed=SEED, rates=list(RATES))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small run to check everything works")
    ap.add_argument("--fresh", action="store_true", help="delete saved runs and start again")
    ap.add_argument("--only", choices=("main", "speed", "season", "hibernate"),
                    help="run one scenario family")
    ap.add_argument("--summary", action="store_true", help="only summarise saved runs")
    ap.add_argument("--workers", type=int, default=0,
                    help="CPU cores to use (default: all but one, at most 8)")
    a = ap.parse_args()
    workers = a.workers or max(1, min(8, (os.cpu_count() or 2) - 1))
    cfg = CONFIG["quick" if a.quick else "full"]

    if a.fresh:
        for f in (cfg["runs"], cfg["done"]):
            if os.path.exists(f):
                os.remove(f)
    done = set()
    if os.path.exists(cfg["done"]):
        with open(cfg["done"]) as f:
            done = set(json.load(f))

    _W.update(load_inputs())
    q, H, terr, provenance = _W["q"], _W["H"], _W["terr"], _W["provenance"]
    print(f"\nq: {provenance}   budget {budget_Wh():.0f} Wh   "
          f"{'QUICK' if a.quick else 'FULL'} run: {cfg['n_targets']} targets x {cfg['n_runs']} runs "
          f"x {len(cfg['times'])} departures, {workers} CPU core(s)\n")

    ident = identity_check(terr, H, q, cfg)
    if not a.summary:
        tasks = [(sc, di, float(hour), cfg["n_targets"], cfg["n_runs"])
                 for sc in scenarios() if not a.only or sc["name"] == a.only
                 for di, hour in enumerate(cfg["times"])
                 if block_key(sc, float(hour)) not in done]
        print(f"  {len(tasks)} blocks to run ({len(done)} already saved)\n")
        t0 = time.time()
        run_all(tasks, cfg, done, workers)
        print(f"\n  simulation time {(time.time() - t0) / 60:.1f} min\n")

    df = pd.read_csv(cfg["runs"], dtype={"rate": str})
    groups = summarise(df)
    print_report(groups)
    print_per_100(groups)
    hl = headline(groups)
    print_headline(hl)
    res = checks(groups, ident)

    out = dict(what="Phase 3 sortie Monte Carlo: coupling-blind vs coupling-aware planner, "
                    "PGDA Site04, VIPER-class rover",
               parameters=collect_params(cfg), q_provenance=provenance,
               sorties=int(len(df)), headline=hl, groups=groups, checks=res,
               limitations=LIMITATIONS)
    with open(cfg["out"], "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {cfg['out']}  ({len(df)} sorties in {cfg['runs']})")


if __name__ == "__main__":
    main()