import sys
sys.path.insert(0, "src/illumination")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy import ndimage

from sun_geometry import sun_position, SYNODIC_HOURS
from horizon_field import horizon_at, DISC_RADIUS_DEG

FIELD = "results/horizon_field.npy"
DT_H = 1.0


def lit_stack(H, dt_h=DT_H):
    hours = np.arange(0.0, SYNODIC_HOURS, dt_h)
    lit = np.empty((len(hours),) + H.shape[:2], dtype=bool)
    for i, hr in enumerate(hours):
        elev, azim = sun_position(hr)
        lit[i] = (elev + DISC_RADIUS_DEG) > horizon_at(H, float(azim))
    return lit


def longest_dark_run(lit):
    n = lit.shape[0]
    run = np.zeros(lit.shape[1:], dtype=np.int32)
    cur = np.zeros_like(run)
    # two passes so a dark stretch that wraps past the end of the month is counted whole
    for k in range(2 * n):
        cur = np.where(~lit[k % n], cur + 1, 0)
        run = np.maximum(run, cur)
    return np.minimum(run, n)


def sunrises(lit):
    return ((~np.roll(lit, 1, axis=0)) & lit).sum(axis=0)


def white_colorbar(cb, label):
    cb.set_label(label, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    cb.outline.set_edgecolor("white")


def plot_longest_dark(days):
    fig, ax = plt.subplots(figsize=(7.2, 7.0), facecolor="black")
    ax.set_facecolor("black")
    im = ax.imshow(days, cmap="inferno_r", vmin=0, vmax=SYNODIC_HOURS / 24)
    ax.set_xticks([]); ax.set_yticks([])
    white_colorbar(plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02),
                   "days of continuous darkness")
    plt.tight_layout()
    plt.savefig("figures/longest_dark.png", dpi=160, facecolor="black")
    plt.close(fig)


def plot_sunrises(rises, always):
    disp = np.clip(rises, 0, 3).astype(np.int16)
    disp[always] = -1
    cols = ["#fcffa4", "#000004", "#5c2a72", "#c43c4e", "#f98e09"]
    labels = ["always lit", "never lit", "1 sunrise, 1 sunset",
              "2 sunrises", "3 or more"]
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], len(cols))

    fig, ax = plt.subplots(figsize=(7.6, 7.0), facecolor="black")
    ax.set_facecolor("black")
    ax.imshow(disp, cmap=ListedColormap(cols), norm=norm, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [Patch(facecolor=cols[i], edgecolor="#666", label=labels[i])
               for i in [1, 2, 3, 4, 0]]
    leg = ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                    frameon=False, fontsize=11, labelspacing=1.0)
    for t in leg.get_texts():
        t.set_color("white")
    plt.tight_layout()
    plt.savefig("figures/sunrises.png", dpi=160, facecolor="black",
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    H = np.load(FIELD)
    lit = lit_stack(H)
    n = lit.shape[0]

    elev = np.array([sun_position(h)[0] for h in np.arange(0.0, SYNODIC_HOURS, DT_H)])
    run = longest_dark_run(lit)
    days = run * DT_H / 24.0
    rises = sunrises(lit)
    never = run >= n
    always = lit.all(axis=0)
    partly = ~never & ~always
    patches, n_patches = ndimage.label(always, structure=np.ones((3, 3)))
    sizes = sorted((int(s) for s in ndimage.sum(always, patches, range(1, n_patches + 1))),
                   reverse=True)

    print(f"sun elevation               : {elev.min():.2f} to {elev.max():.2f} deg, "
          f"{(elev < 0).sum()} hours below horizon")
    print(f"never lit                   : {never.mean():.2%}")
    print(f"median longest dark stretch : {np.median(days[~never]):.2f} days "
          f"(cells that get some sun)")
    print(f"always lit                  : {always.sum()} cells in {n_patches} patches "
          f"{sizes}")
    print(f"partly lit, one sunrise     : {(rises[partly] == 1).mean():.2%}")
    print(f"partly lit, two sunrises    : {(rises[partly] == 2).mean():.2%}")
    print(f"partly lit, three or more   : {(rises[partly] >= 3).mean():.2%}")

    np.save("results/longest_dark_days.npy", days)
    np.save("results/sunrises_per_month.npy", rises)
    plot_longest_dark(days)
    plot_sunrises(rises, always)
    print("\nsaved figures/longest_dark.png and figures/sunrises.png")