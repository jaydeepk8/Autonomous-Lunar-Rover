import numpy as np
import rasterio

PATH = r"data\polar\GroundTruth_RectifiedRangeMaps\Terrain11\PosA_org-XYZ.tif"

with rasterio.open(PATH) as src:
    print("bands:", src.count, "size:", src.width, "x", src.height, "dtype:", src.dtypes)
    xyz = src.read()   # (bands, H, W)

print("array shape:", xyz.shape)

for i, name in enumerate("XYZ"):
    b = xyz[i]
    finite = b[np.isfinite(b) & (b != 0)]
    print(f"{name}: valid {finite.size / b.size:6.1%}  "
          f"min {finite.min():8.3f}  max {finite.max():8.3f}  "
          f"mean {finite.mean():8.3f}")