import glob, os, re, csv
import numpy as np
import cv2
import rasterio
from PIL import Image

POLAR = r"data\polar"
GT_ROOT = r"data\polar\GroundTruth_RectifiedRangeMaps"
OUT = r"results\perception_binned.csv"

TERRAINS = {
    "Terrain01_LargeOnly":   "Terrain01",
    "Terrain03_MeanDistr":   "Terrain03",
    "Terrain06_Smooth":      "Terrain06",
    "Terrain11_FreshCrater": "Terrain11",
    "Terrain12_OldCrater":   "Terrain12",
}

EXP_SUNLIT = (128, 256, 512, 1024)
EXP_NOSUN = (2048, 4096, 8192)

SCALE = 0.5

FX, CX, CY = 1991.49, 966.256, 600.347
BASELINE_MM = 301.556
W, H = 1936, 1216
SIZE = (W, H)

BLACK = 16
Z_MIN, Z_MAX = 1500.0, 7000.0
DARK_MAX_DN = 20.0

MASK_TOP, MASK_BOTTOM = 0.10, 0.30
MASK_SIDE = 0.10

BIN_EDGES = np.arange(2000.0, 6001.0, 500.0)
MIN_GT_PIXELS = 5000

SUN_XY = {
    "Sun_30": (1.422, -3.594),
    "Sun_180": (0.475, 3.544),
    "Sun_270": (-4.109, 0.271),
    "Sun_350": (-0.896, -4.026),
    "NoSun": None,
}
CAM_XY = {"PosA": (-0.15, -3.54), "PosB": (-0.15, -5.50), "PosC": (-3.51, -0.46)}

KNEW = np.array([[FX, 0, CX], [0, FX, CY], [0, 0, 1]], np.float64)

ROI = np.zeros((H, W), bool)
ROI[int(H * MASK_TOP):int(H * (1 - MASK_BOTTOM)),
    int(W * MASK_SIDE):int(W * (1 - MASK_SIDE))] = True

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=128, blockSize=5,
    P1=200, P2=800, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1)


def read_calibration(path):
    vals = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            vals[key.strip()] = [float(x) for x in rest.split()]
    return dict(
        K1=np.array(vals["CAMERA_MATRIX_LEFT"], np.float64).reshape(3, 3),
        d1=np.array(vals["DISTORTION_COEFFICIENTS_LEFT"], np.float64).reshape(1, -1),
        K2=np.array(vals["CAMERA_MATRIX_RIGHT"], np.float64).reshape(3, 3),
        d2=np.array(vals["DISTORTION_COEFFICIENTS_RIGHT"], np.float64).reshape(1, -1),
        R=np.array(vals["ROTATION_MATRIX"], np.float64).reshape(3, 3),
        T=np.array(vals["TRANSLATION_VECTOR"], np.float64).reshape(3, 1),
    )


def build_maps(c):
    R1, R2, _, _, _, _, _ = cv2.stereoRectify(
        c["K1"], c["d1"], c["K2"], c["d2"], SIZE, c["R"], c["T"], alpha=1.0)
    m1x, m1y = cv2.initUndistortRectifyMap(c["K1"], c["d1"], R1, KNEW, SIZE, cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(c["K2"], c["d2"], R2, KNEW, SIZE, cv2.CV_32FC1)
    return m1x, m1y, m2x, m2y


def phase_angle(pos_key, cond):
    sun = SUN_XY.get(cond)
    if sun is None:
        return ""
    c = np.array(CAM_XY[pos_key], float)
    s = np.array(sun, float)
    c /= np.linalg.norm(c)
    s /= np.linalg.norm(s)
    return round(float(np.degrees(np.arccos(np.clip(c @ s, -1, 1)))), 2)


def to8(a):
    lo, hi = np.percentile(a[ROI], [1, 99])
    if hi - lo < 1e-6:
        return None
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


os.makedirs("results", exist_ok=True)
print(f"ROI keeps {ROI.mean():.1%} of the frame; stereo at {SCALE:.0%} resolution")
rows = []

for terrain_dir, gt_terrain in TERRAINS.items():
    root = os.path.join(POLAR, terrain_dir)
    gt_dir = os.path.join(GT_ROOT, gt_terrain)

    if not os.path.isdir(root):
        print(f"{terrain_dir}: folder not found, skipped")
        continue

    calib_path = os.path.join(root, "stereo.calibration")
    if not os.path.exists(calib_path):
        print(f"{terrain_dir}: no stereo.calibration, skipped")
        continue

    calib = read_calibration(calib_path)
    m1x, m1y, m2x, m2y = build_maps(calib)
    print(f"\n{terrain_dir}: fx_left {calib['K1'][0, 0]:.2f}, "
          f"baseline {abs(calib['T'][0, 0]) * 1000:.1f} mm")

    positions = sorted(p for p in os.listdir(root)
                       if os.path.isdir(os.path.join(root, p)) and p.startswith("Pos"))

    for pos in positions:
        pos_key = pos[:4]
        cam_dist_mm = int(pos.split("_")[1])
        lights = pos.split("_")[2]

        gt_path = os.path.join(gt_dir, f"{pos_key}_org-XYZ.tif")
        if not os.path.exists(gt_path):
            print(f"  {pos}: no ground truth, skipped")
            continue
        with rasterio.open(gt_path) as src:
            gtZ = src.read(3)

        gtok = np.isfinite(gtZ) & (gtZ > Z_MIN) & (gtZ < Z_MAX) & ROI
        binidx = np.digitize(gtZ, BIN_EDGES) - 1
        nbins = len(BIN_EDGES) - 1
        rr = np.arange(H)[:, None] * np.ones((1, W))

        for cond in sorted(os.listdir(os.path.join(root, pos))):
            cdir = os.path.join(root, pos, cond)
            if not os.path.isdir(cdir):
                continue

            for lp in sorted(glob.glob(os.path.join(cdir, "CamL_*.png"))):
                exp = int(re.search(r"_(\d+)\.png$", lp).group(1))

                if cond == "NoSun":
                    if exp not in EXP_NOSUN:
                        continue
                elif exp not in EXP_SUNLIT:
                    continue

                rp = glob.glob(os.path.join(cdir, f"CamR_*_{exp:04d}.png"))
                if not rp:
                    continue

                la = np.array(Image.open(lp)).astype(np.float32) - BLACK
                ra = np.array(Image.open(rp[0])).astype(np.float32) - BLACK
                if la.shape != (H, W) or ra.shape != (H, W):
                    continue
                l8, r8 = to8(la), to8(ra)
                if l8 is None or r8 is None:
                    continue

                L = cv2.remap(l8, m1x, m1y, cv2.INTER_LINEAR)
                Rr = cv2.remap(r8, m2x, m2y, cv2.INTER_LINEAR)

                Lh = cv2.resize(L, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
                Rh = cv2.resize(Rr, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
                disp_h = sgbm.compute(Lh, Rh).astype(np.float32) / 16.0
                disp = cv2.resize(disp_h, (W, H), interpolation=cv2.INTER_NEAREST) / SCALE

                depth = np.full_like(disp, np.nan)
                ok = disp > 1.0
                depth[ok] = FX * BASELINE_MM / disp[ok]
                valid = ok & gtok & (depth > Z_MIN) & (depth < Z_MAX)

                base = dict(
                    terrain=terrain_dir, position=pos, pos_key=pos_key,
                    cam_dist_mm=cam_dist_mm, lights=lights,
                    lighting=cond, exposure_ms=exp,
                    phase_deg=phase_angle(pos_key, cond),
                    frame_mean_dn=round(float(la[ROI].mean()), 2),
                    frame_frac_dark=round(float((la[ROI] <= 40).mean()), 4),
                                        frame_frac_sat=round(float((la[ROI] >= 4000).mean()), 4),
                )

                if cond == "NoSun" and lights == "Loff" and la[ROI].mean() > DARK_MAX_DN:
                    continue


                for b in range(nbins):
                    inbin = gtok & (binidx == b)
                    n_gt = int(inbin.sum())
                    if n_gt < MIN_GT_PIXELS:
                        continue

                    v = valid & inbin
                    cov = float(v.sum() / n_gt)
                    rec = dict(base)
                    rec.update(
                        depth_bin_mm=int((BIN_EDGES[b] + BIN_EDGES[b + 1]) / 2),
                        range_m=round((BIN_EDGES[b] + BIN_EDGES[b + 1]) / 2000.0, 3),
                        n_gt=n_gt, n_valid=int(v.sum()),
                        coverage=round(cov, 4),
                        row_centroid=round(float(rr[inbin].mean()), 1),
                        bin_mean_dn=round(float(la[inbin].mean()), 2),
                    )

                    if v.sum() >= 1000 and cov >= 0.20:
                        ae = np.abs(depth[v] - gtZ[v])
                        rec.update(
                            med_err_mm=round(float(np.median(ae)), 2),
                            p75_err_mm=round(float(np.percentile(ae, 75)), 2),
                            p95_err_mm=round(float(np.percentile(ae, 95)), 2))
                    else:
                        rec.update(med_err_mm="", p75_err_mm="", p95_err_mm="")

                    rows.append(rec)

        print(f"  {pos}: done ({len(rows)} rows so far)")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"\nwrote {len(rows)} rows to {OUT}")