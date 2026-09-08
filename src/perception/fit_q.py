import json
import numpy as np
import pandas as pd

CSV = r"results\perception_q.csv"
OUT_JSON = r"results\q_model.json"

a = pd.read_csv(CSV)

# sunlit
s = a[(a.lighting != "NoSun") & a.phase_deg.notna()].copy()
s["range_m"] = s.dist_mm / 1000.0

Xs = np.column_stack([
    s.phase_deg.values,
    s.phase_deg.values ** 2,
    s.range_m.values,
    np.ones(len(s)),
])
bs, *_ = np.linalg.lstsq(Xs, s.q.values, rcond=None)
pred_s = Xs @ bs
r2_s = 1 - np.sum((s.q - pred_s) ** 2) / np.sum((s.q - s.q.mean()) ** 2)
rmse_s = float(np.sqrt(np.mean((s.q - pred_s) ** 2)))
vertex = float(-bs[0] / (2 * bs[1]))

print("SUNLIT REGIME")
print(f"  n            {len(s)}")
print(f"  R2           {r2_s:.3f}")
print(f"  RMSE(q)      {rmse_s:.4f}")
print(f"  vertex       {vertex:.1f} deg")
print(f"  range coef   {bs[2]:+.4f} per m")
print(f"  q = {bs[0]:+.6f}*p {bs[1]:+.6f}*p^2 {bs[2]:+.4f}*r {bs[3]:+.4f}")

# headlight
h = a[(a.lighting == "NoSun") & (a.lights == "Lon")].copy()
h["range_m"] = h.dist_mm / 1000.0
h["inv_r2"] = 1.0 / h.range_m ** 2

Xh = np.column_stack([h.inv_r2.values, np.ones(len(h))])
bh, *_ = np.linalg.lstsq(Xh, h.q.values, rcond=None)
pred_h = Xh @ bh
r2_h = 1 - np.sum((h.q - pred_h) ** 2) / np.sum((h.q - h.q.mean()) ** 2)
rmse_h = float(np.sqrt(np.mean((h.q - pred_h) ** 2)))

print("\nHEADLIGHT REGIME")
print(f"  n            {len(h)}")
print(f"  R2           {r2_h:.3f}")
print(f"  RMSE(q)      {rmse_h:.4f}")
print(f"  q = {bh[0]:+.4f}/r^2 {bh[1]:+.4f}")
for _, r in h.iterrows():
    print(f"    {r.terrain[:9]} {r.pos_key} r={r.range_m:.1f}m  "
          f"q={r.q:.3f}  fit={bh[0]/r.range_m**2 + bh[1]:.3f}")


for thr in (0.5, 0.3, 0.2):
    if bh[0] > 0 and thr > bh[1]:
        rmax = float(np.sqrt(bh[0] / (thr - bh[1])))
        print(f"  q >= {thr:.1f} out to {rmax:.2f} m")
    else:
        print(f"  q >= {thr:.1f} : not resolvable from this fit")

# dark
dark = a[(a.lighting == "NoSun") & (a.lights == "Loff")]
print(f"\nDARK REGIME\n  n {len(dark)}  q = {dark.q.mean():.3f} "
      f"(measured floor, all cases)")

# save
model = {
    "q_definition": "product of log-scaled accuracy and normalised coverage",
    "sunlit": {
        "form": "q = c_p*phase + c_p2*phase^2 + c_r*range_m + c_0",
        "c_p": float(bs[0]), "c_p2": float(bs[1]),
        "c_r": float(bs[2]), "c_0": float(bs[3]),
        "phase_units": "degrees", "vertex_deg": vertex,
        "n": int(len(s)), "r2": float(r2_s), "rmse": rmse_s,
        "valid_range_m": [1.5, 4.0], "valid_phase_deg": [10.0, 175.0],
    },
    "headlight": {
        "form": "q = c_i/range_m^2 + c_0",
        "c_i": float(bh[0]), "c_0": float(bh[1]),
        "n": int(len(h)), "r2": float(r2_h), "rmse": rmse_h,
        "valid_range_m": [1.5, 4.0],
    },
    "dark": {"q": 0.0, "note": "no sun, no lights: measured null, n=%d" % len(dark)},
    "d_of_q": "d = 1 - q, clipped to [0, 1]",
    "provenance": "NASA POLAR stereo dataset, Terrain 11 and 12, 60 conditions",
}

with open(OUT_JSON, "w") as f:
    json.dump(model, f, indent=2)
print(f"\nwrote {OUT_JSON}")