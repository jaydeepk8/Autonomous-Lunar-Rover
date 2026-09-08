import glob
import os
import re
import numpy as np
from PIL import Image

BASE = r"data\polar\Terrain11_FreshCrater\PosA_1500_Loff"
LIGHTING = ["NoSun", "Sun_30", "Sun_180", "Sun_270", "Sun_350"]

for cond in LIGHTING:
    hits = sorted(glob.glob(os.path.join(BASE, cond, "CamL_*.png")))
    exps = sorted({int(re.search(r"_(\d+)\.png$", h).group(1)) for h in hits})
    print(f"\n{cond}  ({len(hits)} left images)  exposures: {exps}")
    for h in hits:
        a = np.array(Image.open(h))
        e = int(re.search(r"_(\d+)\.png$", h).group(1))
        print(f"   {e:5d} ms   mean {a.mean():8.1f}  max {a.max():5d}  "
              f"sat {(a >= 4090).mean():6.3%}  dark {(a <= 50).mean():6.1%}")