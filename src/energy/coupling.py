import json
import os

# tag: spec = published and cited, assumed = swept over the stated range
PARAMS = {
    "lambda0_2023": dict(
        value=1.0 / 1000.0, unit="faults/m", tag="spec",
        source="Lamarre 2023, Acta Astronautica: one fault per 1000 m"),
    "T_rec_2023": dict(
        value=5.0 * 3600.0, unit="s", tag="spec",
        source="Lamarre 2023: 5 h recovery"),
    "lambda0_2024": dict(
        value=1.0 / 5000.0, unit="faults/m", tag="spec",
        source="Lamarre 2024, IEEE Aerospace: one fault per 5000 m"),
    "T_rec_2024": dict(
        value=10.0 * 3600.0, unit="s", tag="spec",
        source="Lamarre 2024: 10 h recovery"),
    "v_drive": dict(
        value=0.8 / 3.6, unit="m/s", tag="spec",
        source="VIPER top speed 0.8 km/h"),
    "v_slow": dict(
        value=0.4 / 3.6, unit="m/s", tag="spec",
        source="VIPER prospecting speed 0.4 km/h"),
    "P_idle": dict(
        value=50.0, unit="W", tag="assumed", sweep=(20.0, 200.0),
        source="ACT/NASA JSC: TMS cuts survival heat from hundreds of W to tens"),
    "P_light": dict(
        value=40.0, unit="W", tag="assumed", sweep=(10.0, 120.0),
        source="no published figure for the VIPER lighting system"),
}

KAPPA_SWEEP = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

Q_MODEL_PATH = os.path.join("results", "q_model.json")

Q_FALLBACK = {
    "sunlit_low": 0.904,
    "sunlit_high": 0.633,
    "headlight": 0.740,
    "dark": 0.033,
}


def load_q():
    if os.path.exists(Q_MODEL_PATH):
        with open(Q_MODEL_PATH) as f:
            m = json.load(f)
        q = {
            "sunlit_low": m["sunlit"]["low_band"]["coverage"],
            "sunlit_high": m["sunlit"]["high_band"]["coverage"],
            "headlight": m["headlight"]["coverage"],
            "dark": m["dark"]["coverage"],
        }
        return q, m, f"read from {Q_MODEL_PATH}"

    print("! results/q_model.json not found, using hardcoded Phase 1 constants")
    print("! run src/perception/fit_final.py to regenerate it\n")
    return dict(Q_FALLBACK), None, "HARDCODED FALLBACK"


def per_terrain_q(m):
    """Terrains carrying both a headlight and a dark measurement."""
    h = m["headlight"]["per_terrain"]
    d = m["dark"]["per_terrain"]
    return {t: (h[t], d[t]) for t in sorted(set(h) & set(d))}


def p(name):
    return PARAMS[name]["value"]


def fault_rate(q, kappa, lambda0):
    d = min(max(1.0 - q, 0.0), 1.0)
    return lambda0 * (1.0 + kappa * d)


def breakeven_headlight_power(q_dark, q_head, kappa, lambda0, T_rec, v, P_idle):
    dlam = fault_rate(q_dark, kappa, lambda0) - fault_rate(q_head, kappa, lambda0)
    return dlam * v * T_rec * P_idle


def segment_energy(L, q, kappa, lambda0, T_rec, v, P_base, P_drive,
                   P_light, P_idle):
    """Expected energy in J to traverse L metres in one perception mode."""
    t = L / v
    return (P_base + P_drive + P_light) * t + \
        fault_rate(q, kappa, lambda0) * L * T_rec * P_idle


def report_params():
    print("=" * 74)
    print("PARAMETERS")
    print("=" * 74)
    for name, d in PARAMS.items():
        sweep = f"  sweep {d['sweep']}" if "sweep" in d else ""
        print(f"  {name:14s} {d['value']:>10.5g} {d['unit']:<10s} "
              f"[{d['tag']}]{sweep}")
        print(f"                 {d['source']}")
    print()


def report_modes(q):
    print("=" * 74)
    print("PERCEPTION MODES  (q from Phase 1)")
    print("=" * 74)
    print(f"  {'mode':<14s} {'q':>8s} {'d = 1-q':>9s}")
    for k in ["sunlit_low", "sunlit_high", "headlight", "dark"]:
        print(f"  {k:<14s} {q[k]:8.3f} {1.0 - q[k]:9.3f}")
    print()


def report_breakeven(q):
    print("=" * 74)
    print("HEADLIGHT BREAK-EVEN POWER")
    print("=" * 74)
    print(f"  assumed draw {p('P_light'):.0f} W, sweep {PARAMS['P_light']['sweep']}\n")

    for label, lam_key, trec_key in [
        ("Lamarre 2023  (1/1000 m, 5 h)", "lambda0_2023", "T_rec_2023"),
        ("Lamarre 2024  (1/5000 m, 10 h)", "lambda0_2024", "T_rec_2024"),
    ]:
        print(f"  {label}")
        print(f"    {'kappa':>7s} {'P_be (W)':>11s} {'verdict':>16s}")
        for kappa in KAPPA_SWEEP:
            P_be = breakeven_headlight_power(
                q["dark"], q["headlight"], kappa,
                p(lam_key), p(trec_key), p("v_drive"), p("P_idle"))
            if P_be <= 0.0:
                verdict = "never worth it"
            elif P_be > p("P_light"):
                verdict = "lights ON"
            else:
                verdict = "lights OFF"
            print(f"    {kappa:7.2f} {P_be:11.1f} {verdict:>16s}")
        print()

    print("  kappa = 0 gives exactly zero: with no coupling lights never pay.")
    print("  Every non-zero row is the coupling changing a decision.\n")


def report_idle_sensitivity(q):
    print("=" * 74)
    print("SENSITIVITY: break-even vs idle draw  (Lamarre 2023, kappa = 1.0)")
    print("=" * 74)
    lo, hi = PARAMS["P_idle"]["sweep"]
    print(f"    {'P_idle (W)':>11s} {'P_be (W)':>11s}")
    for P_idle in [lo, 50.0, 100.0, 150.0, hi]:
        P_be = breakeven_headlight_power(
            q["dark"], q["headlight"], 1.0,
            p("lambda0_2023"), p("T_rec_2023"), p("v_drive"), P_idle)
        print(f"    {P_idle:11.0f} {P_be:11.1f}")
    print()


def report_per_terrain(m, q):
    """
    Phase 1 limitation 5: headlight coverage varies 0.50 to 0.87 by terrain.
    Pooling q hides that the decision itself flips between terrains.
    """
    print("=" * 74)
    print("PER-TERRAIN BREAK-EVEN  (Lamarre 2023, P_idle = 50 W)")
    print("=" * 74)

    pt = per_terrain_q(m)
    kappas = [0.25, 0.5, 1.0]

    print(f"  {'terrain':24s} {'q_head':>7s} {'q_dark':>7s}", end="")
    for k in kappas:
        print(f" {'k=' + str(k):>9s}", end="")
    print()

    for t, (qh, qd) in pt.items():
        print(f"  {t:24s} {qh:7.3f} {qd:7.3f}", end="")
        for k in kappas:
            P_be = breakeven_headlight_power(
                qd, qh, k, p("lambda0_2023"), p("T_rec_2023"),
                p("v_drive"), p("P_idle"))
            print(f" {P_be:9.1f}", end="")
        print()

    qh, qd = q["headlight"], q["dark"]
    print(f"  {'POOLED':24s} {qh:7.3f} {qd:7.3f}", end="")
    for k in kappas:
        P_be = breakeven_headlight_power(
            qd, qh, k, p("lambda0_2023"), p("T_rec_2023"),
            p("v_drive"), p("P_idle"))
        print(f" {P_be:9.1f}", end="")
    print()

    print(f"\n  At the assumed {p('P_light'):.0f} W draw the pooled model and the "
          f"per-terrain models")
    print("  disagree at low kappa: pooled says OFF while the two crater "
          "terrains")
    print("  say ON. A single global q cannot carry the headlight decision.")
    print("  Terrain 1 is absent: no NoSun capture in the source dataset.\n")


if __name__ == "__main__":
    q, model, provenance = load_q()
    print(f"\nq provenance: {provenance}\n")
    report_params()
    report_modes(q)
    report_breakeven(q)
    report_idle_sensitivity(q)
    if model is not None:
        report_per_terrain(model, q)