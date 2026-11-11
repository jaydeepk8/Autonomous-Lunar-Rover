import sys
sys.path.insert(0, "src/illumination")
sys.path.insert(0, "src/energy")

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy import sparse

from sun_geometry import sun_position
from horizon_field import load_dem, load_field, sun_elevation, disc_fraction, WIDE
from irradiance import solar_power
from loads import p as load_p, drive_power
from coupling import load_q, PARAMS as FAULT

SLOPE_5M = "data/Site04_final_adj_5mpp_slp.tif"
SLOPE_ERR_5M = "data/Site04_final_adj_5mpp_slperr.tif"

PARAMS = {
    "max_slope": dict(
        value=25.0, unit="deg", tag="assumed", sweep=(20.0, 30.0),
        source="VIPER climbs 15 deg with ease and 25 to 30 deg if needed"),
    "max_steep_share": dict(
        value=0.5, unit="-", tag="assumed", sweep=(0.25, 0.75),
        source="the rover can steer round isolated steep 5 m patches inside a "
               "40 m cell, but not through a cell that is mostly steep"),
    "v_eff": dict(
        value=0.8 / 3.6, unit="m/s", tag="assumed", sweep=(0.05, 0.02),
        source="average speed including stops for imaging, planning and contact; "
               "VIPER plans at about 1 cm/s speed made good (Ennico-Smith et al. 2023). "
               "Default equals driving speed, as in Phase 2"),
    "phase_low": dict(
        value=105.0, unit="deg", tag="spec",
        source="results/q_model.json: highest phase angle in the low band"),
    "phase_high": dict(
        value=170.0, unit="deg", tag="spec",
        source="results/q_model.json: lowest phase angle in the high band"),
}

STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def p(name):
    return PARAMS[name]["value"]


def shift(a, dr, dc, fill=np.nan):
    # out[r, c] = a[r + dr, c + dc]
    out = np.full(a.shape, fill, dtype=np.result_type(a, type(fill)))
    h, w = a.shape
    out[max(-dr, 0):h - max(dr, 0), max(-dc, 0):w - max(dc, 0)] = \
        a[max(dr, 0):h - max(-dr, 0), max(dc, 0):w - max(-dc, 0)]
    return out


def slope_5m(xs, ys):
    if not os.path.exists(SLOPE_5M):
        return None
    res = xs[1] - xs[0]
    dst_tf = from_origin(xs[0] - res / 2, ys[0] + res / 2, res, res)
    with rasterio.open(WIDE) as w:
        crs = w.crs
    with rasterio.open(SLOPE_5M) as s:
        src = s.read(1, masked=True).astype(np.float32).filled(np.nan)
        src_tf, src_crs = s.transform, s.crs

    raw = src
    err = None
    if os.path.exists(SLOPE_ERR_5M):
        with rasterio.open(SLOPE_ERR_5M) as e:
            if e.shape == src.shape and e.transform == src_tf:
                err = e.read(1, masked=True).astype(np.float32).filled(np.nan)
    if err is not None:
        # random DEM error always steepens slopes; remove it in quadrature
        t2 = np.tan(np.radians(src)) ** 2 - np.tan(np.radians(err)) ** 2
        src = np.degrees(np.arctan(np.sqrt(np.maximum(t2, 0.0)))).astype(np.float32)

    def to_grid(a, how):
        out = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)
        reproject(a, out, src_transform=src_tf, src_crs=src_crs, src_nodata=np.nan,
                  dst_transform=dst_tf, dst_crs=crs, dst_nodata=np.nan, resampling=how)
        return out

    def steep_share(a):
        return to_grid(np.where(np.isfinite(a), (a > p("max_slope")).astype(np.float32),
                                np.nan), Resampling.average)

    return dict(share=steep_share(src),
                share_raw=steep_share(raw),
                mean=to_grid(src, Resampling.average),
                max=to_grid(src, Resampling.max),
                err=None if err is None else to_grid(err, Resampling.average))


def terrain():
    dem, res, r0, r1, c0, c1, tb, xs, ys = load_dem()
    z = dem[r0:r1, c0:c1]
    s5 = slope_5m(xs, ys)
    passable = np.isfinite(z)
    if s5 is not None:
        passable &= np.nan_to_num(s5["share"], nan=1.0) <= p("max_steep_share")

    steps = []
    for dr, dc in STEPS:
        L = res * np.hypot(dr, dc)
        dz = shift(z, dr, dc) - z
        grade = np.degrees(np.arctan2(dz, L))
        valid = (np.isfinite(dz) & passable & shift(passable, dr, dc, False)
                 & (np.abs(grade) <= p("max_slope")))
        steps.append(dict(dr=dr, dc=dc, L=np.hypot(L, np.nan_to_num(dz)),
                          grade=np.nan_to_num(grade), valid=valid,
                          az=np.degrees(np.arctan2(dc, dr)) % 360.0))
    return dict(z=z, xs=xs, ys=ys, passable=passable, s5=s5, steps=steps)


def q_sunlit(phase_deg, q):
    w = np.clip((phase_deg - p("phase_low")) / (p("phase_high") - p("phase_low")), 0.0, 1.0)
    return q["sunlit_low"] * (1.0 - w) + q["sunlit_high"] * w


def stop_seconds_per_metre(v):
    return max(1.0 / p("v_eff") - 1.0 / v, 0.0)


def energy_per_metre(P_idle, P_drive, P_light, P_sun, q, kappa, lam0, T_rec, v):
    lam = lam0 * (1.0 + kappa * np.clip(1.0 - q, 0.0, 1.0))
    move = np.maximum(P_idle + P_drive + P_light - P_sun, 0.0) / v / 3600.0
    # stopped time when average speed is below driving speed: idle load, lights off
    move = move + np.maximum(P_idle - P_sun, 0.0) * stop_seconds_per_metre(v) / 3600.0
    fault = lam * T_rec * np.maximum(P_idle - P_sun, 0.0) / 3600.0
    return move, fault


def edge_costs(terr, H, hour, kappa, q, rate="2023"):
    xs, ys = terr["xs"], terr["ys"]
    lam0, T_rec = FAULT[f"lambda0_{rate}"]["value"], FAULT[f"T_rec_{rate}"]["value"]
    v = load_p("v_drive")

    elev, azim = sun_position(hour)
    e = sun_elevation(float(elev), float(azim), xs, ys)
    f = disc_fraction(H, e, float(azim))
    P_sun = solar_power(H, xs, ys, float(elev), float(azim))
    lit = f > 0.0
    P_idle = load_p("P_avionics") + np.where(lit, load_p("P_heater_sun"),
                                             load_p("P_heater_shadow"))

    costs = np.full((len(STEPS),) + lit.shape, np.inf)
    lights = np.zeros((len(STEPS),) + lit.shape, dtype=bool)
    for k, s in enumerate(terr["steps"]):
        P_drive = drive_power(s["grade"])
        phase = 180.0 - np.abs((s["az"] - float(azim) + 180.0) % 360.0 - 180.0)
        m, fl = energy_per_metre(P_idle, P_drive, 0.0, P_sun, q_sunlit(phase, q),
                                 kappa, lam0, T_rec, v)
        e_lit = m + fl
        m, fl = energy_per_metre(P_idle, P_drive, 0.0, 0.0, q["dark"],
                                 kappa, lam0, T_rec, v)
        e_off = m + fl
        m, fl = energy_per_metre(P_idle, P_drive, load_p("P_light"), 0.0, q["headlight"],
                                 kappa, lam0, T_rec, v)
        e_on = m + fl
        per_m = np.where(lit, e_lit, np.minimum(e_on, e_off))
        lights[k] = ~lit & (e_on < e_off) & s["valid"]
        costs[k] = np.where(s["valid"], per_m * s["L"], np.inf)
    return costs, lights, lit


def to_graph(costs):
    n_steps, h, w = costs.shape
    idx = np.arange(h * w).reshape(h, w)
    rows, cols, data = [], [], []
    for k, (dr, dc) in enumerate(STEPS):
        ok = np.isfinite(costs[k])
        rows.append(idx[ok])
        cols.append(shift(idx, dr, dc, -1)[ok])
        data.append(costs[k][ok])
    rows, cols, data = map(np.concatenate, (rows, cols, data))
    return sparse.csr_matrix((data, (rows, cols)), shape=(h * w, h * w))


def report_terrain(terr):
    print("=" * 74)
    print("TERRAIN")
    print("=" * 74)
    n = terr["passable"].size
    print(f"  grid              : {terr['z'].shape[0]} x {terr['z'].shape[1]} cells, "
          f"{terr['xs'][1] - terr['xs'][0]:.0f} m")
    lim = p("max_slope")
    print(f"  slope limit       : {lim:.0f} deg")
    if terr["s5"] is None:
        print(f"  5 m slope map     : NOT FOUND at {SLOPE_5M}; using 40 m grades only")
    else:
        s5 = terr["s5"]
        if s5["err"] is None:
            print(f"  5 m slope error   : NOT USED ({SLOPE_ERR_5M} missing or not aligned)")
        else:
            print(f"  5 m slope error   : median {np.nanmedian(s5['err']):.1f} deg, "
                  f"removed from each 5 m slope")
            print(f"  blocked cells     : {np.nanmean(s5['share_raw'] > p('max_steep_share')):.2%} "
                  f"before the correction, "
                  f"{np.nanmean(s5['share'] > p('max_steep_share')):.2%} after")
        print(f"  5 m slope per cell: mean {np.nanmedian(s5['mean']):.1f} deg (median cell), "
              f"steepest {np.nanmedian(s5['max']):.1f} deg (median cell)")
        print(f"\n  share of cells blocked under different rules:")
        rules = [
            ("any 5 m patch steeper than the limit", s5["max"] > lim),
            ("over 25% of the cell steeper", s5["share"] > 0.25),
            ("over 50% of the cell steeper", s5["share"] > 0.50),
            ("over 75% of the cell steeper", s5["share"] > 0.75),
            ("cell's mean 5 m slope over the limit", s5["mean"] > lim),
        ]
        for name, mask in rules:
            print(f"    {name:<40s} {np.nanmean(mask):7.2%}")
        print(f"  rule in use: over {p('max_steep_share'):.0%} of the cell steeper\n")
    print(f"  impassable cells  : {(~terr['passable']).sum()} ({(~terr['passable']).mean():.2%})")
    allowed = sum(s["valid"].sum() for s in terr["steps"])
    inside = sum(np.isfinite(shift(terr["z"], s["dr"], s["dc"])).sum() for s in terr["steps"])
    print(f"  moves allowed     : {allowed} of {inside} ({allowed / inside:.2%})\n")


def report_per_km(q):
    print("=" * 74)
    print("EXPECTED ENERGY PER KM, FLAT GROUND  (Wh/km; driving + faults)")
    print("=" * 74)
    v, P_drive = load_p("v_drive"), float(drive_power(0.0))
    lit_idle = load_p("P_avionics") + load_p("P_heater_sun")
    dark_idle = load_p("P_avionics") + load_p("P_heater_shadow")
    rows = [
        ("sunlit, sun behind (q 0.904)", lit_idle, 0.0, 0.0, q["sunlit_low"]),
        ("sunlit, into the sun (q 0.633)", lit_idle, 0.0, 0.0, q["sunlit_high"]),
        ("shadow, lights off (q 0.033)", dark_idle, 0.0, 0.0, q["dark"]),
        ("shadow, lights on (q 0.740)", dark_idle, load_p("P_light"), 0.0, q["headlight"]),
    ]
    for rate in ("2023", "2024"):
        lam0, T_rec = FAULT[f"lambda0_{rate}"]["value"], FAULT[f"T_rec_{rate}"]["value"]
        print(f"  Lamarre {rate} fault rate")
        print(f"    {'situation':<32s} {'kappa 0':>17s} {'kappa 1':>17s}")
        for name, P_idle, P_l, P_sun, qq in rows:
            out = []
            for kappa in (0.0, 1.0):
                m, fl = energy_per_metre(P_idle, P_drive, P_l, P_sun, qq, kappa, lam0, T_rec, v)
                out.append(f"{m * 1000:5.0f} + {fl * 1000:5.0f}")
            print(f"    {name:<32s} {out[0]:>17s} {out[1]:>17s}")
        print()
    print("  Sunlit rows ignore the sunlight itself, to isolate the effect of q.")
    print("  On the map, sunlight is subtracted from both parts.\n")


def report_map(terr, H, q, hour=350.0):
    print("=" * 74)
    print(f"MAP AT t = {hour:.0f} h  (Lamarre 2023 rate)")
    print("=" * 74)
    print(f"  {'kappa':>6s} {'sunlit Wh/km':>13s} {'shadow Wh/km':>13s} {'lights on':>10s}")
    for kappa in (0.0, 0.25, 0.5, 1.0, 2.0):
        costs, lights, lit = edge_costs(terr, H, hour, kappa, q)
        per_km = []
        for mask in (lit, ~lit):
            vals = [costs[k][mask & np.isfinite(costs[k])] / terr["steps"][k]["L"][mask & np.isfinite(costs[k])]
                    for k in range(len(STEPS))]
            per_km.append(np.median(np.concatenate(vals)) * 1000)
        dark_moves = sum((np.isfinite(costs[k]) & ~lit).sum() for k in range(len(STEPS)))
        print(f"  {kappa:6.2f} {per_km[0]:13.0f} {per_km[1]:13.0f} "
              f"{lights.sum() / max(dark_moves, 1):10.1%}")
    print(f"\n  sunlit cells at this moment: {lit.mean():.1%}")
    g = to_graph(costs)
    print(f"  graph: {g.shape[0]} cells, {g.nnz} allowed moves\n")


if __name__ == "__main__":
    q, _, provenance = load_q()
    H, _, _ = load_field()
    terr = terrain()
    print(f"\nq: {provenance}\n")
    report_terrain(terr)
    report_per_km(q)
    report_map(terr, H, q)