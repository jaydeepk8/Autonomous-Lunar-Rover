import glob, os, re, csv
import numpy as np
import cv2
import rasterio
from PIL import Image

POLAR = r"data\polar"
GT_ROOT = r"data\polar\GroundTruth_RectifiedRangeMaps"
OUT = r"results\perception_labels.csv"

TERRAINS = {
    "Terrain11_FreshCrater": "Terrain11",
    "Terrain12_OldCrater":   "Terrain12",
}

FX, CX, CY = 1991.49, 966.256, 600.347
BASELINE_MM = 301.556
SIZE = (1936, 1216)
BLACK = 16
Z_MIN, Z_MAX = 1500.0, 7000.0

SUN_XY = {
    "Sun_30": (1.422, -3.594),
    "Sun_180": (0.475, 3.544),
    "Sun_270": (-4.109, 0.271),
    "Sun_350": (-0.896, -4.026),
    "NoSun": None,
}
CAM_XY = {"PosA": (-0.15, -3.54), "PosB": (-0.15, -5.50), "PosC": (-3.51, -0.46)}

K1 = np.array([[2068.44, 0, 964.405], [0, 2064.29, 593.512], [0, 0, 1]], np.float64)
d1 = np.array([-0.108441, 0.158389, 0.000537106, -0.00127904], np.float64).reshape(1, 4)
K2 = np.array([[2065.6, 0, 952.098], [0, 2061.9, 606.559], [0, 0, 1]], np.float64)
d2 = np.array([-0.114041, 0.178219, -0.000146877, -0.00112736], np.float64).reshape(1, 4)
R = np.array([[0.999997, 0.00172602, 0.0019164],
              [-0.00172807, 0.999998, 0.00106908],
              [-0.00191455, -0.00107239, 0.999998]], np.float64)
T = np.array([[-0.301556], [0.000599], [0.001489]], np.float64)
KNEW = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1]], np.float64)

R1, R2, _, _, _, _, _ = cv2.stereoRectify(K1, d1, K2, d2, SIZE, R, T, alpha=1.0)
m1x, m1y = cv2.initUndistortRectifyMap(K1, d1, R1, KNEW, SIZE, cv2.CV_32FC1)
m2x, m2y = cv2.initUndistortRectifyMap(K2, d2, R2, KNEW, SIZE, cv2.CV_32FC1)

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=256, blockSize=5,
    P1=200, P2=800, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1)


def phase_angle(pos_key, cond):
    sun = SUN_XY.get(cond)
    if sun is None:
        return ""
    c = np.array(CAM_XY[pos_key], float)
    s = np.array(sun, float)
    c /= np.linalg.norm(c)
    s /= np.linalg.norm(s)
    return round(float(np.degrees(np.arccos(np.clip(c @ s, -1, 1)))), 2)


def load_raw(path):
    return np.array(Image.open(path)).astype(np.float32) - BLACK


def to8(a):
    lo, hi = np.percentile(a, [1, 99])
    if hi - lo < 1e-6:
        return None
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


os.makedirs("results", exist_ok=True)
rows = []

for terrain_dir, gt_terrain in TERRAINS.items():
    root = os.path.join(POLAR, terrain_dir)
    gt_dir = os.path.join(GT_ROOT, gt_terrain)

    if not os.path.isdir(root):
        print(f"{terrain_dir}: folder not found, skipped")
        continue

    positions = sorted(p for p in os.listdir(root)
                       if os.path.isdir(os.path.join(root, p)) and p.startswith("Pos"))

    for pos in positions:
        pos_key = pos[:4]
        dist_mm = int(pos.split("_")[1])
        lights = pos.split("_")[2]

        gt_path = os.path.join(gt_dir, f"{pos_key}_org-XYZ.tif")
        if not os.path.exists(gt_path):
            print(f"{terrain_dir}/{pos}: no ground truth, skipped")
            continue
        with rasterio.open(gt_path) as src:
            gtZ = src.read(3)
        gtok = np.isfinite(gtZ) & (gtZ > Z_MIN) & (gtZ < Z_MAX)

        for cond in sorted(os.listdir(os.path.join(root, pos))):
            cdir = os.path.join(root, pos, cond)
            if not os.path.isdir(cdir):
                continue

            for lp in sorted(glob.glob(os.path.join(cdir, "CamL_*.png"))):
                exp = int(re.search(r"_(\d+)\.png$", lp).group(1))
                rp = glob.glob(os.path.join(cdir, f"CamR_*_{exp:04d}.png"))
                if not rp:
                    continue

                la, ra = load_raw(lp), load_raw(rp[0])
                l8, r8 = to8(la), to8(ra)

                rec = dict(
                    terrain=terrain_dir,
                    position=pos, pos_key=pos_key, dist_mm=dist_mm, lights=lights,
                    lighting=cond, exposure_ms=exp,
                    phase_deg=phase_angle(pos_key, cond),
                    mean_dn=round(float(la.mean()), 2),
                    std_dn=round(float(la.std()), 2),
                    frac_dark=round(float((la <= 40).mean()), 4),
                    frac_sat=round(float((la >= 4000).mean()), 4),
                )

                if l8 is None or r8 is None:
                    rec.update(coverage="", med_err_mm="", p75_err_mm="",
                               p95_err_mm="", n_valid=0)
                    rows.append(rec)
                    continue

                L = cv2.remap(l8, m1x, m1y, cv2.INTER_LINEAR)
                Rr = cv2.remap(r8, m2x, m2y, cv2.INTER_LINEAR)
                disp = sgbm.compute(L, Rr).astype(np.float32) / 16.0

                depth = np.full_like(disp, np.nan)
                ok = disp > 1.0
                depth[ok] = FX * BASELINE_MM / disp[ok]
                valid = ok & gtok & (depth > Z_MIN) & (depth < Z_MAX)

                if valid.sum() / gtok.sum() < 0.20:
                    rec.update(coverage=round(float(valid.sum() / gtok.sum()), 4),
                               med_err_mm="", p75_err_mm="", p95_err_mm="",
                               n_valid=int(valid.sum()))
                else:
                    ae = np.abs(depth[valid] - gtZ[valid])
                    rec.update(
                        coverage=round(float(valid.sum() / gtok.sum()), 4),
                        med_err_mm=round(float(np.median(ae)), 2),
                        p75_err_mm=round(float(np.percentile(ae, 75)), 2),
                        p95_err_mm=round(float(np.percentile(ae, 95)), 2),
                        n_valid=int(valid.sum()))

                rows.append(rec)

        print(f"{terrain_dir}/{pos}: done ({len(rows)} rows so far)")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nwrote {len(rows)} rows to {OUT}")