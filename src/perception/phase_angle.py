import numpy as np

SUN = {
    "Sun_30":  (1.422, -3.594),
    "Sun_180": (0.475,  3.544),
    "Sun_270": (-4.109, 0.271),
    "Sun_350": (-0.896, -4.026),
}

CAM = {
    "PosA_1500_Loff": (-0.15, -3.54),
    "PosB_4000_Loff": (-0.15, -5.50),
    "PosC_1500_Loff": (-3.51, -0.46),
}

# median absolute depth error, mm, from the last run
MED = {
    ("PosA_1500_Loff", "Sun_30"): 21.8,
    ("PosA_1500_Loff", "Sun_180"): 23.0,
    ("PosA_1500_Loff", "Sun_270"): 12.2,
    ("PosA_1500_Loff", "Sun_350"): 29.3,
    ("PosB_4000_Loff", "Sun_30"): 48.7,
    ("PosB_4000_Loff", "Sun_180"): 59.2,
    ("PosB_4000_Loff", "Sun_270"): 54.9,
    ("PosB_4000_Loff", "Sun_350"): 61.8,
    ("PosC_1500_Loff", "Sun_30"): 15.0,
    ("PosC_1500_Loff", "Sun_180"): 18.4,
    ("PosC_1500_Loff", "Sun_270"): 28.4,
    ("PosC_1500_Loff", "Sun_350"): 14.3,
}

COVER = {
    ("PosA_1500_Loff", "Sun_30"): 76.5,
    ("PosA_1500_Loff", "Sun_180"): 46.3,
    ("PosA_1500_Loff", "Sun_270"): 69.7,
    ("PosA_1500_Loff", "Sun_350"): 67.4,
    ("PosB_4000_Loff", "Sun_30"): 85.3,
    ("PosB_4000_Loff", "Sun_180"): 55.6,
    ("PosB_4000_Loff", "Sun_270"): 70.3,
    ("PosB_4000_Loff", "Sun_350"): 85.1,
    ("PosC_1500_Loff", "Sun_30"): 68.3,
    ("PosC_1500_Loff", "Sun_180"): 75.8,
    ("PosC_1500_Loff", "Sun_270"): 62.8,
    ("PosC_1500_Loff", "Sun_350"): 71.3,
}


def phase(cam, sun):
    """Angle at the terrain origin between the light and the camera, degrees."""
    c = np.array(cam) / np.linalg.norm(cam)
    s = np.array(sun) / np.linalg.norm(sun)
    return float(np.degrees(np.arccos(np.clip(c @ s, -1, 1))))


rows = []
print(f"{'position':16s} {'cond':9s} {'phase':>7s} {'range':>7s} "
      f"{'med':>7s} {'cover':>7s}")
for pos, cam in CAM.items():
    rng = np.linalg.norm(cam)
    for cond, sun in SUN.items():
        ph = phase(cam, sun)
        m, cv = MED[(pos, cond)], COVER[(pos, cond)]
        rows.append((ph, rng, m, cv))
        print(f"{pos:16s} {cond:9s} {ph:7.1f} {rng:7.2f} {m:7.1f} {cv:7.1f}")

a = np.array(rows)
print("\ncorrelations (Pearson):")
print(f"  phase vs median error   {np.corrcoef(a[:,0], a[:,2])[0,1]:+.3f}")
print(f"  phase vs coverage       {np.corrcoef(a[:,0], a[:,3])[0,1]:+.3f}")
print(f"  range vs median error   {np.corrcoef(a[:,1], a[:,2])[0,1]:+.3f}")

near = a[a[:,1] < 4.0]
print("\nnear positions only (PosA, PosC):")
print(f"  phase vs median error   {np.corrcoef(near[:,0], near[:,2])[0,1]:+.3f}")
print(f"  phase vs coverage       {np.corrcoef(near[:,0], near[:,3])[0,1]:+.3f}")