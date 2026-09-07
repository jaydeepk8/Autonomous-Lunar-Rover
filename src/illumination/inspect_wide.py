import rasterio
import numpy as np

WIDE = "data/LDEM_80S_40MPP_ADJ.tiff"
TILE = "data/Site04_final_adj_5mpp_surf.tif"
PSR  = "data/LPSR_80S_20MPP_ADJ.tiff"

for name, path in [("WIDE", WIDE), ("TILE", TILE), ("PSR", PSR)]:
    with rasterio.open(path) as src:
        print(f"\n--- {name} ---")
        print("size:", src.width, "x", src.height)
        print("res (m):", src.res)
        print("bounds (m):", src.bounds)
        print("dtype:", src.dtypes[0])
        print("nodata:", src.nodata)
        print("crs same as tile:", src.crs.to_string()[:60])

with rasterio.open(TILE) as t, rasterio.open(WIDE) as w:
    tb = t.bounds

    r0, c0 = w.index(tb.left,  tb.top)
    r1, c1 = w.index(tb.right, tb.bottom)
    print("\ntile occupies wide-DEM rows", r0, "to", r1,
          "cols", c0, "to", c1)
    print("that is", abs(r1 - r0), "x", abs(c1 - c0), "cells at 40 m")

    x, y = (tb.left + tb.right) / 2, (tb.bottom + tb.top) / 2
    tr, tc = t.index(x, y)
    wr, wc = w.index(x, y)
    z_tile = t.read(1)[tr, tc]
    z_wide = w.read(1, window=((wr, wr + 1), (wc, wc + 1)))[0, 0]
    print(f"\nat centre ({x:.0f}, {y:.0f}) m:")
    print("  tile elevation:", float(z_tile))
    print("  wide elevation:", float(z_wide))
    print("  difference (m):", float(z_tile - z_wide))