import sys
sys.path.insert(0, "src/illumination")
sys.path.insert(0, "src/energy")
sys.path.insert(0, "src/planning")

import time
from contextlib import contextmanager
import numpy as np

from horizon_field import load_field
from edges import terrain
from coupling import load_q, PARAMS as FAULT
from loads import idle_load
from planner import Planner, budget_Wh, cell
from execute import World, fly, PARAMS

HOUR = 350.0
RATES = ("2023", "2024")
N_TARGETS = 50
N_RUNS = 300
SEED = 7
TRACE_OUT = "results/sortie_trace.npz"


@contextmanager
def no_faults():
    old = {r: FAULT[f"lambda0_{r}"]["value"] for r in RATES}
    for r in RATES:
        FAULT[f"lambda0_{r}"]["value"] = 0.0
    try:
        yield
    finally:
        for r in RATES:
            FAULT[f"lambda0_{r}"]["value"] = old[r]


def accepted_targets(planner, dep, n, seed=SEED):
    dark = planner.terr["passable"].ravel() & ~dep["lit"].ravel()
    ok = np.flatnonzero(dark & planner.accepts(dep))
    rng = np.random.default_rng(seed)
    return rng.choice(ok, size=min(n, ok.size), replace=False) if ok.size else ok


def example_target(blind, aware):
    # same choice as check_planner.py
    d0, d1 = blind.departure(HOUR), aware.departure(HOUR)
    dark = blind.terr["passable"].ravel() & ~d0["lit"].ravel()
    rt0, rt1 = blind.round_trip(d0), aware.round_trip(d1)
    lost = np.flatnonzero(dark & (rt0 <= budget_Wh()) & (rt1 > budget_Wh()))
    if lost.size == 0:
        return None
    return int(lost[np.argsort(rt0[lost])[lost.size // 2]])


def report_params():
    print("=" * 74)
    print("PARAMETERS")
    print("=" * 74)
    for k, d in PARAMS.items():
        print(f"  {k:12s} {d['value']:>8.4g} {d['unit']:<3s} [{d['tag']}]")
        print(f"               {d['source']}")
    print(f"\n  departure at t = {HOUR:.0f} h, full battery, budget {budget_Wh():.0f} Wh\n")


def check_plan_equals_reality(terr, H, q):
    print("=" * 74)
    print(f"1. PLAN = REALITY  (sun frozen, fault rate zero, {N_TARGETS} targets)")
    print("=" * 74)
    with no_faults():
        pl = Planner("blind", terr, H, q, 0.0)
        dep = pl.departure(HOUR)
        world = World(terr, H, q, frozen=HOUR)
        targets = accepted_targets(pl, dep, N_TARGETS)
        errs, back = [], 0
        for t in targets:
            r = fly(world, pl, dep, int(t), 0.0)
            errs.append(abs(r["ledger"] - r["predicted"]) / r["predicted"])
            back += r["outcome"] == "within budget" and r["reached"]
    worst = max(errs) if errs else np.nan
    good = len(errs) > 0 and worst < 1e-5 and back == len(errs)
    print(f"  worst gap between simulated energy and planner prediction: {worst:.1e}")
    print(f"  reached target and returned within budget: {back} of {len(errs)}")
    print("  (gaps under 1e-5 are float32 rounding in partly lit cells)")
    print(f"  {'PASS' if good else 'FAIL'}\n")
    return good, targets


def check_moving_sun(terr, H, q, targets):
    print("=" * 74)
    print(f"2. MOVING SUN ALONE  (fault rate zero, same {len(targets)} targets)")
    print("=" * 74)
    with no_faults():
        pl = Planner("blind", terr, H, q, 0.0)
        dep = pl.departure(HOUR)
        frozen, moving = World(terr, H, q, frozen=HOUR), World(terr, H, q)
        d, hrs, over = [], [], 0
        for t in targets:
            a = fly(frozen, pl, dep, int(t), 0.0)
            b = fly(moving, pl, dep, int(t), 0.0)
            d.append((b["ledger"] - a["ledger"]) / a["ledger"])
            hrs.append(b["hours"])
            over += b["ledger"] > budget_Wh()
    d = np.array(d)
    print(f"  energy change from sun movement: median {np.median(d):+.2%}, "
          f"range {d.min():+.2%} to {d.max():+.2%}")
    print(f"  sortie length: median {np.median(hrs):.1f} h, longest {max(hrs):.1f} h")
    print(f"  pushed over budget by sun movement alone: {over} of {len(d)}\n")
    return True


def check_forced_fault(terr, H, q, target):
    print("=" * 74)
    print("3. ONE FORCED FAULT AT THE TARGET  (sun frozen, no random faults)")
    print("=" * 74)
    ok_all = True
    for rate in RATES:
        pl = Planner("blind", terr, H, q, 0.0, rate)
        dep = pl.departure(HOUR)
        world = World(terr, H, q, rate, frozen=HOUR)
        a = fly(world, pl, dep, target, 0.0)
        b = fly(world, pl, dep, target, 0.0, forced_target=1)
        extra = b["drawdown"] - a["drawdown"]
        expect = idle_load(shadow=True) * world.T_rec
        good = abs(extra - expect) < 1e-6 * expect
        ok_all &= good
        print(f"  Lamarre {rate}: extra energy {extra:7.1f} Wh, expected {expect:7.1f} Wh "
              f"({idle_load(shadow=True):.0f} W x {world.T_rec:.0f} h)  {'PASS' if good else 'FAIL'}")
        print(f"                 drawdown {a['drawdown']:.0f} -> {b['drawdown']:.0f} Wh, "
              f"outcome '{b['outcome']}'")
    print()
    return ok_all


def check_turn_back(terr, H, q, target):
    print("=" * 74)
    print("4. TURN-BACK RULE  (blind planner, Lamarre 2023, kappa_true 1, moving sun)")
    print("=" * 74)
    pl = Planner("blind", terr, H, q, 0.0)
    dep = pl.departure(HOUR)
    world = World(terr, H, q)
    from planner import outbound_path
    n_moves = len(outbound_path(dep, target)) - 1
    at = max(1, n_moves // 4)
    print(f"  example target {cell(target, terr['passable'].shape)}, {n_moves} moves out, "
          f"planner predicts {dep['out'][target] + dep['home'][target]:.0f} Wh")
    print(f"  fault forced after move {at}\n")

    closed = fly(world, pl, dep, target, 1.0, forced_out={at})
    opened = fly(world, pl, dep, target, 1.0, forced_out={at}, turn_back=False)
    for dsc in closed["decisions"]:
        print(f"  t+{dsc['t'] - HOUR:5.1f} h  move {dsc['move']:3d}  charge {dsc['soc']:6.0f} Wh  "
              f"still needs {dsc['need']:6.0f} Wh  allowed {dsc['allowance']:6.0f} Wh  -> "
              f"{'TURN BACK' if dsc['turn_back'] else 'continue'}")
    print()
    for name, r in (("with turn-back", closed), ("without turn-back", opened)):
        print(f"  {name:<18s} reached target {str(r['reached']):<5s} drawdown {r['drawdown']:6.0f} Wh "
              f"({r['drawdown'] / budget_Wh():5.0%} of budget)  {r['outcome']}")
    good = closed["turned_back"] and closed["drawdown"] <= opened["drawdown"]
    print(f"\n  {'PASS' if good else 'CHECK'}: the rule "
          f"{'turned back and saved energy' if good else 'did not turn back here'}\n")
    return good


def check_random_faults(terr, H, q, target):
    print("=" * 74)
    print(f"5. RANDOM FAULTS  (blind planner, example target, {N_RUNS} runs each, moving sun)")
    print("=" * 74)
    ok_all = True
    print(f"  {'rate':>5s} {'kappa':>6s} {'faults':>7s} {'expected':>9s} {'z':>6s}   "
          f"{'within':>7s} {'turned':>7s} {'over':>6s} {'strand':>7s}  {'ms/run':>7s}")
    for rate in RATES:
        pl = Planner("blind", terr, H, q, 0.0, rate)
        dep = pl.departure(HOUR)
        world = World(terr, H, q, rate)
        for kappa in (0.0, 1.0):
            t0 = time.time()
            rs = [fly(world, pl, dep, target, kappa, rng=np.random.default_rng([SEED, i]))
                  for i in range(N_RUNS)]
            ms = (time.time() - t0) / N_RUNS * 1000
            f = np.mean([r["faults"] for r in rs])
            e = np.mean([r["expected_faults"] for r in rs])
            z = (f - e) / np.sqrt(max(e, 1e-12) / N_RUNS)
            good = abs(z) < 3.0
            ok_all &= good
            share = {o: np.mean([r["outcome"] == o for r in rs])
                     for o in ("within budget", "turned back", "over budget", "stranded")}
            print(f"  {rate:>5s} {kappa:6.1f} {f:7.2f} {e:9.2f} {z:+6.2f}   "
                  f"{share['within budget']:7.1%} {share['turned back']:7.1%} "
                  f"{share['over budget']:6.1%} {share['stranded']:7.1%}  {ms:7.1f}  "
                  f"{'PASS' if good else 'FAIL'}")
    print("\n  faults vs expected: mean count against mean of (rate x distance); |z| < 3 passes")
    print("  outcome shares are a preview on one target, not the Phase 3 result\n")
    return ok_all


def show_trace(terr, H, q, target):
    print("=" * 74)
    print("6. ONE SORTIE STEP BY STEP  (blind, Lamarre 2023, kappa_true 1, first run with a fault)")
    print("=" * 74)
    pl = Planner("blind", terr, H, q, 0.0)
    dep = pl.departure(HOUR)
    world = World(terr, H, q)
    for i in range(N_RUNS):
        r = fly(world, pl, dep, target, 1.0, rng=np.random.default_rng([SEED, i]), record=True)
        if r["faults"] > 0:
            break
    shape = terr["passable"].shape
    tr = r["trace"]
    print(f"  run {i}: {r['outcome']}, {r['faults']} fault(s), {r['hours']:.1f} h, "
          f"drawdown {r['drawdown']:.0f} Wh\n")
    print(f"  {'t+h':>6s} {'cell':>11s} {'charge Wh':>10s} {'sun':>5s} {'lights':>7s}  event")
    for j, (t, idx, soc, lit, lights, ev) in enumerate(tr):
        if ev or j % 10 == 0:
            print(f"  {t - HOUR:6.2f} {str(cell(idx, shape)):>11s} {soc:10.0f} "
                  f"{'lit' if lit else 'dark':>5s} {'on' if lights else 'off':>7s}  {ev}")
    np.savez(TRACE_OUT, t=np.array([x[0] for x in tr]), idx=np.array([x[1] for x in tr]),
             soc=np.array([x[2] for x in tr]), lit=np.array([x[3] for x in tr]),
             lights=np.array([x[4] for x in tr]), event=np.array([x[5] for x in tr]),
             target=target, hour=HOUR, shape=shape)
    print(f"\n  every 10th step and all events shown; saved {TRACE_OUT}\n")


def run(terr, H, q):
    report_params()
    checks = []
    good, targets = check_plan_equals_reality(terr, H, q)
    checks.append(("plan = reality", good))
    check_moving_sun(terr, H, q, targets)

    blind = Planner("blind", terr, H, q, 0.0)
    aware = Planner("aware", terr, H, q, 1.0)
    target = example_target(blind, aware)
    if target is None:
        print("no example target at this moment; stopping\n")
        return
    checks.append(("forced fault adds idle x T_rec", check_forced_fault(terr, H, q, target)))
    checks.append(("turn-back rule acts", check_turn_back(terr, H, q, target)))
    checks.append(("fault count matches rate", check_random_faults(terr, H, q, target)))
    show_trace(terr, H, q, target)

    print("=" * 74)
    print("SUMMARY")
    print("=" * 74)
    for name, good in checks:
        print(f"  {name:<32s} {'PASS' if good else 'FAIL'}")
    print()


if __name__ == "__main__":
    q, _, provenance = load_q()
    H, _, _ = load_field()
    terr = terrain()
    print(f"\nq: {provenance}\ngrid {terr['passable'].shape}\n")
    run(terr, H, q)