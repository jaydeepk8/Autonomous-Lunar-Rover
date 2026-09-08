import numpy as np
import pandas as pd

CSV = r"results\perception_labels.csv"
OUT = r"results\perception_q.csv"
COVERAGE_FLOOR = 0.20

d = pd.read_csv(CSV)

d.loc[d.coverage < COVERAGE_FLOOR,
      ["med_err_mm", "p75_err_mm", "p95_err_mm"]] = np.nan

d["usable"] = 1 - d.frac_dark - d.frac_sat

idx = d.groupby(["terrain", "position", "lighting"])["usable"].idxmax()
a = d.loc[idx].copy()

lo, hi = 10.0, 200.0
err = a.med_err_mm.clip(lo, hi)
q_acc = 1 - (np.log(err) - np.log(lo)) / (np.log(hi) - np.log(lo))

q_cov = (a.coverage / a.coverage.max()).clip(0, 1)

q_acc = q_acc.fillna(0.0)
q_cov = q_cov.where(a.med_err_mm.notna(), 0.0)

a["q_acc"] = q_acc.round(4)
a["q_cov"] = q_cov.round(4)
a["q"] = (q_acc * q_cov).round(4)
a["d"] = (1 - a["q"]).round(4)

cols = ["terrain", "pos_key", "dist_mm", "lights", "lighting", "exposure_ms",
        "phase_deg", "coverage", "med_err_mm", "q_acc", "q_cov", "q", "d"]
a = a[cols].sort_values(["terrain", "pos_key", "lights", "lighting"])
a.to_csv(OUT, index=False)

print(a.to_string(index=False))
print(f"\n{len(a)} conditions -> {OUT}")
print(f"q range {a.q.min():.3f} to {a.q.max():.3f}")