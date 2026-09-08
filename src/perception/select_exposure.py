import glob
import os
import re
import numpy as np
from PIL import Image

BASE = r"data\polar\Terrain11_FreshCrater\PosA_1500_Loff"
LIGHTING = ["NoSun", "Sun_30", "Sun_180", "Sun_270", "Sun_350"]

BLACK = 16
LOW = 40
HIGH = 4000 

for cond in LIGHTING:
    hits = sorted(glob.glob(os.path.join(BASE, cond, "CamL_*.png")))
    best = None
    print(f"\n{cond}")
    for h in hits:
        e = int(re.search(r"_(\d+)\.png$", h).group(1))
        a = np.array(Image.open(h)).astype(np.int32) - BLACK
        good = float(((a > LOW) & (a < HIGH)).mean())
        print(f"   {e:5d} ms   usable {good:6.1%}")
        if best is None or good > best[1]:
            best = (e, good)
    print(f"   -> selected {best[0]} ms  ({best[1]:.1%} usable)")