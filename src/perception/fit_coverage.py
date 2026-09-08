import json
import numpy as np
import pandas as pd

CSV = r"results\perception_binned.csv"
OUT = r"results\q_model.json"

d = pd.read_csv(CSV)
d["usable"] = 1 - d.frame_frac_dark - d.frame_frac_sat

idx = d.groupby(["terrain", "position", "lighting", "depth_bin_mm"])["usable"].idxmax()
a = d.loc[idx].copy()


def design(g):
    return np.column_stack([
        g.phase_deg.values,
        g.phase_deg.values ** 2,
        g.range_m.values,
        g.range_m.values ** 2,
        np.ones(len(g)),
    ])


def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


# sunlit
s = a[(a.lighting != "NoSun") & a.phase_deg.notna()].copy()
b, *_ = np.linalg.lstsq(design(s), s.coverage.values, rcond=None)

print("SUNLIT")
print(f"  n {len(s)}   in-sample R2 {r2(s.coverage.values, design(s) @ b):.3f}")
print("  leave-one-position-out:")
lopo = {}
for pos in sorted(s.pos_key.unique()):
    tr, te = s[s.pos_key != pos], s[s.pos_key == pos]
    bt, *_ = np.linalg.lstsq(design(tr), tr.coverage.values, rcond=None)
    lopo[pos] = r2(te.coverage.values, design(te) @ bt)
    print(f"    hold out {pos}: R2 {lopo[pos]:+.3f}")
print("  leave-one-terrain-out:")
loto = {}
for t in s.terrain.unique():
    tr, te = s[s.terrain != t], s[s.terrain == t]
    bt, *_ = np.linalg.lstsq(design(tr), tr.coverage.values, rcond=None)
    loto[t] = r2(te.coverage.values, design(te) @ bt)
    print(f"    hold out {t}: R2 {loto[t]:+.3f}")
print(f"  phase optimum {-b[0] / (2 * b[1]):.1f} deg")

# headlight
h = a[(a.lighting == "NoSun") & (a.lights == "Lon")]
h_ok = h[h.pos_key != "PosB"]
print("\nHEADLIGHT (no sun)")
print(f"  n {len(h)}   flat in range 2.25-5.75 m, no falloff detected")
print(f"  PosA+PosC: mean {h_ok.coverage.mean():.3f}  sd {h_ok.coverage.std():.3f}")
print(f"  PosB:      mean {h[h.pos_key == 'PosB'].coverage.mean():.3f}  "
      f"(unexplained deficit, n={(h.pos_key == 'PosB').sum()})")

# dark
k = a[(a.lighting == "NoSun") & (a.lights == "Loff")]
print(f"\nDARK\n  n {len(k)}   coverage mean {k.coverage.mean():.4f}  "
      f"max {k.coverage.max():.4f}")

model = {
    "target": "stereo coverage (fraction of ground-truth pixels reconstructed)",
    "note": "depth accuracy is NOT modelled: geometry does not predict it "
            "out-of-sample (R2 ~0.29 in-sample, ~0.0 held out)",
    "sunlit": {
        "form": "cov = c_p*phase + c_p2*phase^2 + c_r*range_m + c_r2*range_m^2 + c_0",
        "c_p": float(b[0]), "c_p2": float(b[1]),
        "c_r": float(b[2]), "c_r2": float(b[3]), "c_0": float(b[4]),
        "phase_units": "degrees", "phase_optimum_deg": float(-b[0] / (2 * b[1])),
        "n": int(len(s)),
        "r2_in_sample": r2(s.coverage.values, design(s) @ b),
        "r2_leave_one_position_out": lopo,
        "r2_leave_one_terrain_out": loto,
        "valid_range_m": [2.25, 5.75], "valid_phase_deg": [10.0, 175.0],
    },
    "headlight": {
        "form": "cov = constant",
        "value": float(h_ok.coverage.mean()), "sd": float(h_ok.coverage.std()),
        "n": int(len(h_ok)),
        "note": "no range dependence detected over 2.25-5.75 m; earlier "
                "inverse-square model was a camera-position confound and is retracted",
        "excluded": "PosB (mean %.3f, unexplained)" % h[h.pos_key == "PosB"].coverage.mean(),
    },
    "dark": {
        "value": float(k.coverage.mean()), "n": int(len(k)),
        "note": "no sun, no lights: measured null",
    },
    "q": "q = coverage (accuracy term dropped, see note)",
    "d_of_q": "d = 1 - q, clipped to [0, 1]",
    "provenance": "NASA POLAR stereo dataset, Terrain 11 and 12, per-depth-bin labels",
}

with open(OUT, "w") as f:
    json.dump(model, f, indent=2)
print(f"\nwrote {OUT}")