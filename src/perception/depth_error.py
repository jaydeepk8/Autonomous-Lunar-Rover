import glob, os
import numpy as np
import cv2
import rasterio
from PIL import Image

ROOT = r"data\polar\Terrain11_FreshCrater"
GT_DIR = r"data\polar\GroundTruth_RectifiedRangeMaps\Terrain11"

POSITIONS = {
    "PosA_1500_Loff": "PosA_org-XYZ.tif",
    "PosB_4000_Loff": "PosB_org-XYZ.tif",
    "PosC_1500_Loff": "PosC_org-XYZ.tif",
}

LIGHTING = ["Sun_30", "Sun_180", "Sun_270", "Sun_350"]
SELECTED = {"Sun_30": 256, "Sun_180": 128, "Sun_270": 1024, "Sun_350": 256}

FX, CX, CY = 1991.49, 966.256, 600.347
BASELINE_MM = 301.556
SIZE = (1936, 1216)
BLACK = 16
Z_MIN, Z_MAX = 1500.0, 7000.0

K1 = np.array([[2068.44, 0, 964.405],
               [0, 2064.29, 593.512],
               [0, 0, 1]], np.float64)
d1 = np.array([-0.108441, 0.158389, 0.000537106, -0.00127904], np.float64).reshape(1, 4)

K2 = np.array([[2065.6, 0, 952.098],
               [0, 2061.9, 606.559],
               [0, 0, 1]], np.float64)
d2 = np.array([-0.114041, 0.178219, -0.000146877, -0.00112736], np.float64).reshape(1, 4)

R = np.array([[0.999997, 0.00172602, 0.0019164],
              [-0.00172807, 0.999998, 0.00106908],
              [-0.00191455, -0.00107239, 0.999998]], np.float64)
T = np.array([[-0.301556], [0.000599], [0.001489]], np.float64)

KNEW = np.array([[FX, 0, CX],
                 [0, FX, CY],
                 [0, 0, 1]], np.float64)

R1, R2, _, _, _, _, _ = cv2.stereoRectify(K1, d1, K2, d2, SIZE, R, T, alpha=1.0)
m1x, m1y = cv2.initUndistortRectifyMap(K1, d1, R1, KNEW, SIZE, cv2.CV_32FC1)
m2x, m2y = cv2.initUndistortRectifyMap(K2, d2, R2, KNEW, SIZE, cv2.CV_32FC1)

sgbm = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=256, blockSize=5,
    P1=8 * 5 * 5, P2=32 * 5 * 5, uniquenessRatio=10,
    speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1)


def load8(pos, cond, cam, exposure):
    hits = glob.glob(os.path.join(ROOT, pos, cond, f"{cam}_*_{exposure:04d}.png"))
    if not hits:
        return None
    a = np.array(Image.open(hits[0])).astype(np.float32) - BLACK
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


print(f"{'position':16s} {'cond':10s} {'exp':>6s}  {'cover':>7s} "
      f"{'med':>8s} {'p75':>8s} {'p95':>8s}")

for pos, gt_name in POSITIONS.items():
    if not os.path.isdir(os.path.join(ROOT, pos)):
        print(f"{pos:16s} -- folder not found, skipped")
        continue

    with rasterio.open(os.path.join(GT_DIR, gt_name)) as src:
        gtZ = src.read(3)
    gtok = np.isfinite(gtZ) & (gtZ > Z_MIN) & (gtZ < Z_MAX)

    for cond in LIGHTING:
        exp = SELECTED[cond]
        li = load8(pos, cond, "CamL", exp)
        ri = load8(pos, cond, "CamR", exp)
        if li is None or ri is None:
            print(f"{pos:16s} {cond:10s} {exp:6d}  no image at this exposure")
            continue

        L = cv2.remap(li, m1x, m1y, cv2.INTER_LINEAR)
        Rr = cv2.remap(ri, m2x, m2y, cv2.INTER_LINEAR)

        disp = sgbm.compute(L, Rr).astype(np.float32) / 16.0

        depth = np.full_like(disp, np.nan)
        ok = disp > 1.0
        depth[ok] = FX * BASELINE_MM / disp[ok]

        valid = ok & gtok & (depth > Z_MIN) & (depth < Z_MAX)
        if valid.sum() < 1000:
            print(f"{pos:16s} {cond:10s} {exp:6d}  too few valid pixels")
            continue

        ae = np.abs(depth[valid] - gtZ[valid])
        print(f"{pos:16s} {cond:10s} {exp:6d} {valid.sum() / gtok.sum():7.1%} "
              f"{np.median(ae):8.1f} {np.percentile(ae, 75):8.1f} "
              f"{np.percentile(ae, 95):8.1f}")
    print()