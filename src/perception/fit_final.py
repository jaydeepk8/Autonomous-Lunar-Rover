import json
import numpy as np
import pandas as pd

CSV = r"results\perception_binned.csv"
OUT = r"results\q_model.json"

LOW_MAX = 105.0    # highest phase angle observed in the low band
HIGH_MIN = 170.0   # lowest phase angle observed in the high band

d = pd.read_csv(CSV)
d["usable"] = 1 - d.frame_frac_dark - d.frame_frac_sat

# one frame per condition per depth bin: emulate rover auto-exposure
idx = d.groupby(["terrain", "position", "lighting", "depth_bin_mm"])["usable"].idxmax()
a = d.loc[idx].copy()

s = a[(a.lighting != "NoSun") & a.phase_deg.notna()].copy()
low = s[s.phase_deg <= LOW_MAX]
high = s[s.phase_deg >= HIGH_MIN]
head = a[(a.lighting == "NoSun") & (a.lights == "Lon")]
dark = a[(a.lighting == "NoSun") & (a.lights == "Loff")]

terrains = sorted(a.terrain.unique())
print(f"terrains: {len(terrains)}")
for t in terrains:
    print(f"  {t}")


def band_report(name, g):
    print(f"\n{name}:  n {len(g)}  mean {g.coverage.mean():.3f}  sd {g.coverage.std():.3f}")
    for t, gg in g.groupby("terrain"):
        print(f"    {t:24s} n {len(gg):3d}  mean {gg.coverage.mean():.3f}")
    return {
        "coverage": float(g.coverage.mean()), "sd": float(g.coverage.std()),
        "n": int(len(g)),
        "per_terrain": {t: round(float(gg.coverage.mean()), 4)
                        for t, gg in g.groupby("terrain")},
    }


# is the low band flat, per terrain?
print("\nLOW BAND flatness (linear trend within band):")
slopes = {}
for t, g in low.groupby("terrain"):
    sl = float(np.polyfit(g.phase_deg, g.coverage, 1)[0])
    slopes[t] = round(sl, 6)
    print(f"    {t:24s} slope {sl:+.5f}/deg")
sl_all = float(np.polyfit(low.phase_deg, low.coverage, 1)[0])
print(f"    {'ALL':24s} slope {sl_all:+.5f}/deg")

low_d = band_report("SUNLIT low  (phase <= 105)", low)
high_d = band_report("SUNLIT high (phase >= 170)", high)
head_d = band_report("HEADLIGHT (no sun, lights on)", head)
dark_d = band_report("DARK (no sun, no lights)", dark)

low_d.update(phase_deg_max=LOW_MAX,
             phases_observed=sorted(float(p) for p in low.phase_deg.unique()),
             trend_slope_per_deg=round(sl_all, 6),
             trend_slope_per_terrain=slopes,
             note="no trend with phase angle within this band, on any terrain")
high_d.update(phase_deg_min=HIGH_MIN,
              phases_observed=sorted(float(p) for p in high.phase_deg.unique()))

model = {
    "target": "stereo coverage: fraction of ground-truth pixels reconstructed",
    "q": "q = coverage",
    "d_of_q": "d = 1 - q, clipped to [0, 1]",
    "terrains": terrains,
    "sunlit": {
        "form": "piecewise constant over two observed phase-angle bands",
        "low_band": low_d,
        "high_band": high_d,
        "unobserved_band_deg": [LOW_MAX, HIGH_MIN],
        "unobserved_note": "the dataset contains no lighting geometry between "
                           "105 and 174 deg. The shape of the transition is "
                           "unknown. Interpolate linearly only as an explicit "
                           "modelling assumption, not as a fitted result.",
    },
    "headlight": head_d,
    "dark": dark_d,
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
        "Headlight coverage varies strongly by terrain (0.50 to 0.87), much "
        "more than sunlit coverage does.",
        "Terrain 3 PosC has a light leak in the source data: no-sun no-light "
        "frames read 99-398 DN against 2-7 DN elsewhere, scaling linearly with "
        "exposure. Those frames are excluded at extraction.",
        "Central-ROI mask applied (top 10%, bottom 30%, 10% each side) per the "
        "dataset readme, to exclude the sandbox lip.",
        "Stereo computed at half resolution for tractability; coverage is a "
        "ratio and is insensitive to this, absolute depth error is not.",
                "Terrain 1 has no NoSun capture at all in the source dataset, so the "
        "headlight and dark regimes rest on four terrains, not five. It also "
        "has no PosC_1500_Lon capture.",
    ],
    "provenance": "NASA POLAR stereo dataset, per-depth-bin labels, "
                  "5 terrains: rock fields, featureless, and two craters",
}

with open(OUT, "w") as f:
    json.dump(model, f, indent=2)
print(f"\nwrote {OUT}")