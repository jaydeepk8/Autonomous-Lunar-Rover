import rasterio
import numpy as np

PATH = "data/Site04_final_adj_5mpp_surf.tif"

with rasterio.open(PATH) as src:
    print("size (w x h):", src.width, "x", src.height)
    print("bands:", src.count)
    print("dtype:", src.dtypes[0])
    print("crs:", src.crs)
    print("bounds (m):", src.bounds)
    print("pixel size (m):", src.res)
    print("nodata:", src.nodata)

    dem = src.read(1)

valid = dem[np.isfinite(dem)]
print("\nelevation min (m):", float(valid.min()))
print("elevation max (m):", float(valid.max()))
print("elevation mean (m):", float(valid.mean()))
print("relief (m):", float(valid.max() - valid.min()))