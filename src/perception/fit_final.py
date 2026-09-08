import json
import numpy as np
import pandas as pd

CSV = r"results\perception_binned.csv"
OUT = r"results\q_model.json"

LOW_MAX = 105.0    # highest phase angle observed in the low band
HIGH_MIN = 170.0   # lowest phase angle observed in the high band

d = pd.read_csv(CSV)
d["usable"] = 1 - d.frame_frac_dark - d.frame_frac_sat
idx = d.groupby(["terrain", "position", "lighting", "depth_bin_mm"])["usable"].idxmax()
a = d.loc[idx].copy()

s = a[(a.lighting != "NoSun") & a.phase_deg.notna()].copy()
low = s[s.phase_deg <= LOW_MAX]
high = s[s.phase_deg >= HIGH_MIN]

print("SUNLIT, two observed bands")
print(f"  low  phase <= {LOW_MAX:.0f} deg : n {len(low):3d}  "
      f"mean {low.coverage.mean():.3f}  sd {low.coverage.std():.3f}  "
      f"phases {sorted(low.phase_deg.unique())}")
print(f"  high phase >= {HIGH_MIN:.0f} deg : n {len(high):3d}  "
      f"mean {high.coverage.mean():.3f}  sd {high.coverage.std():.3f}  "
      f"phases {sorted(high.phase_deg.unique())}")
print(f"  UNOBSERVED between {LOW_MAX:.0f} and {HIGH_MIN:.0f} deg")

# is the low band flat? test against a linear trend within it
xl = low.phase_deg.values
yl = low.coverage.values
bl = np.polyfit(xl, yl, 1)
r2l = 1 - np.sum((yl - np.polyval(bl, xl)) ** 2) / np.sum((yl - yl.mean()) ** 2)
print(f"\n  low band linear trend: slope {bl[0]:+.5f}/deg  R2 {r2l:.4f} "
      f"-> {'flat' if r2l < 0.05 else 'NOT flat, investigate'}")

# does the two-band split hold across terrains and positions?
print("\n  band means by terrain:")
for k, g in s.groupby("terrain"):
    lo = g[g.phase_deg <= LOW_MAX].coverage.mean()
    hi = g[g.phase_deg >= HIGH_MIN].coverage.mean()
    print(f"    {k:24s} low {lo:.3f}  high {hi:.3f}  drop {lo - hi:+.3f}")
print("  band means by position:")
for k, g in s.groupby("pos_key"):
    lo = g[g.phase_deg <= LOW_MAX]
    hi = g[g.phase_deg >= HIGH_MIN]
    lo_m = lo.coverage.mean() if len(lo) else float("nan")
    hi_m = hi.coverage.mean() if len(hi) else float("nan")
    print(f"    {k}  low {lo_m:.3f} (n={len(lo)})  high {hi_m:.3f} (n={len(hi)})")

h = a[(a.lighting == "NoSun") & (a.lights == "Lon")]
h_ok = h[h.pos_key != "PosB"]
k = a[(a.lighting == "NoSun") & (a.lights == "Loff")]
print(f"\nHEADLIGHT  n {len(h_ok)}  mean {h_ok.coverage.mean():.3f}  sd {h_ok.coverage.std():.3f}")
print(f"DARK       n {len(k)}  mean {k.coverage.mean():.4f}")

model = {
    "target": "stereo coverage: fraction of ground-truth pixels reconstructed",
    "q": "q = coverage",
    "d_of_q": "d = 1 - q, clipped to [0, 1]",
    "sunlit": {
        "form": "piecewise constant over two observed phase-angle bands",
        "low_band": {
            "phase_deg_max": LOW_MAX,
            "coverage": float(low.coverage.mean()),
            "sd": float(low.coverage.std()), "n": int(len(low)),
            "phases_observed": sorted(float(p) for p in low.phase_deg.unique()),
            "note": "no trend with phase angle within this band",
        },
        "high_band": {
            "phase_deg_min": HIGH_MIN,
            "coverage": float(high.coverage.mean()),
            "sd": float(high.coverage.std()), "n": int(len(high)),
            "phases_observed": sorted(float(p) for p in high.phase_deg.unique()),
        },
        "unobserved_band_deg": [LOW_MAX, HIGH_MIN],
        "unobserved_note": "the dataset contains no lighting geometry between "
                           "105 and 174 deg. The shape of the transition is "
                           "unknown. Interpolate linearly only as an explicit "
                           "modelling assumption, not as a fitted result.",
    },
    "headlight": {
        "form": "constant", "coverage": float(h_ok.coverage.mean()),
        "sd": float(h_ok.coverage.std()), "n": int(len(h_ok)),
        "note": "no range dependence detected over 2.75-5.75 m",
        "excluded": "PosB, unexplained deficit",
    },
    "dark": {"coverage": float(k.coverage.mean()), "n": int(len(k)),
             "note": "no sun, no lights: measured null"},
    "limitations": [
        "Sunlit model is two measured constants, not a curve. Only 12 distinct "
        "sun-camera geometries exist, 10 of them below 105 deg phase.",
        "The high band rests on two phase angles (173.9, 174.8), both from the "
        "Sun_180 lamp, though seen from two camera positions.",
        "No range term. In-frame depth correlates with image row at -0.954 for "
        "a fixed downward-pitched stereo pair, so range and viewing geometry "
        "cannot be separated in this dataset.",
        "Depth accuracy is not modelled; geometry does not predict it "
        "out-of-sample.",
        "Central-ROI mask applied (top 10%, bottom 30%, 10% each side) per the "
        "dataset readme, to exclude the sandbox lip.",
        "Two terrains, both craters.",
    ],
    "provenance": "NASA POLAR stereo dataset, Terrain 11 and 12, per-depth-bin labels",
}

with open(OUT, "w") as f:
    json.dump(model, f, indent=2)
print(f"\nwrote {OUT}")