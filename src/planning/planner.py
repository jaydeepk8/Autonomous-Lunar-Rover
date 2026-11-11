import sys
sys.path.insert(0, "src/illumination")
sys.path.insert(0, "src/energy")

from collections import OrderedDict
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from sun_geometry import sun_position
from loads import p as load_p, idle_load, drive_power
from edges import edge_costs, shift, STEPS
from coupling import breakeven_headlight_power, PARAMS as FAULT
from battery import p as batt_p

PARAMS = {
    "budget": dict(
        value=0.30, unit="-", tag="assumed",
        source="share of the battery one sortie may use; same as the Phase 2 headline"),
    "replan_h": dict(
        value=2.0, unit="h", tag="assumed", sweep=(1.0, 4.0),
        source="how often the planner refreshes its sun; the sun moves about "
               "0.5 deg per hour"),
}

NO_PRED = -9999
_GRAPHS = {}


def p(name):
    return PARAMS[name]["value"]


def budget_Wh():
    return p("budget") * batt_p("E_batt")


def charge_power():
    # sunlight must cover the parked, sunlit load
    return load_p("P_avionics") + load_p("P_heater_sun")


def graph_layout(terr):
    """Which cells connect to which never changes, so build the network layout once."""
    key = id(terr)
    if key in _GRAPHS:
        return _GRAPHS[key]
    h, w = terr["passable"].shape
    n = h * w
    idx = np.arange(n).reshape(h, w)
    masks = [s["valid"] for s in terr["steps"]]
    rows = np.concatenate([idx[m] for m in masks])
    cols = np.concatenate([shift(idx, dr, dc, -1)[m] for (dr, dc), m in zip(STEPS, masks)])
    order = np.arange(1, rows.size + 1, dtype=np.float64)

    def layout(r, c):
        g = sparse.csr_matrix((order, (r, c)), shape=(n, n))
        return g.data.astype(np.int64) - 1, g.indices, g.indptr

    _GRAPHS[key] = dict(n=n, masks=masks, fwd=layout(rows, cols), bwd=layout(cols, rows))
    return _GRAPHS[key]


class Planner:
    """What one planner believes: costs at its own kappa, sun frozen at the hour it plans."""

    def __init__(self, name, terr, H, q, kappa, rate="2023", cache_size=40):
        self.name, self.terr, self.H, self.q = name, terr, H, q
        self.kappa, self.rate = float(kappa), rate
        self.shape = terr["passable"].shape
        self.size = cache_size
        self.views = OrderedDict()
        self.departs = {}

        self.layout = graph_layout(terr)
        self.P_drive = [drive_power(s["grade"]) for s in terr["steps"]]

        lam0 = FAULT[f"lambda0_{rate}"]["value"]
        T_rec = FAULT[f"T_rec_{rate}"]["value"]
        self.P_be = breakeven_headlight_power(q["dark"], q["headlight"], self.kappa, lam0,
                                              T_rec, load_p("v_drive"), idle_load(shadow=True))
        # same choice edges.py makes per move: lights on in shadow when they pay
        self.lights = bool(self.P_be > load_p("P_light"))

    def bucket(self, hour):
        return float(np.floor(hour / p("replan_h")) * p("replan_h"))

    def _build(self, hour, forward):
        terr, lay = self.terr, self.layout
        costs, _, lit, P_sun = edge_costs(terr, self.H, hour, self.kappa, self.q, self.rate,
                                          P_drive=self.P_drive, return_sun=True)
        data = np.concatenate([costs[k][m] for k, m in enumerate(lay["masks"])]) + 1e-9
        del costs
        n = lay["n"]

        def graph(side):
            perm, ind, ptr = lay[side]
            return sparse.csr_matrix((data[perm], ind, ptr), shape=(n, n))

        _, azim = sun_position(hour)
        charge = terr["passable"] & (P_sun >= charge_power())
        src = np.flatnonzero(charge)

        def tree(g):
            if src.size == 0:
                return np.full(n, np.inf), np.full(n, NO_PRED, dtype=np.int32)
            d, pr, _ = dijkstra(g, indices=src, min_only=True, return_predecessors=True)
            return d, pr.astype(np.int32)

        # home: cheapest energy from each cell to the nearest charging cell
        home, home_pred = tree(graph("bwd"))
        v = dict(hour=hour, azim=float(azim), n_charge=int(src.size))
        if not forward:
            v["home"], v["home_pred"] = home.astype(np.float32), home_pred
            return v
        v["home"], v["home_pred"] = home, home_pred
        v["out"], v["out_pred"] = tree(graph("fwd"))
        v["lit"], v["charge"], v["P_sun"] = lit, charge, P_sun
        return v

    def view_at(self, hour):
        hour = float(hour)
        if hour in self.views:
            self.views.move_to_end(hour)
            return self.views[hour]
        v = self._build(hour, forward=False)
        self.views[hour] = v
        if len(self.views) > self.size:
            self.views.popitem(last=False)
        return v

    def view(self, hour):
        return self.view_at(self.bucket(hour))

    def departure(self, hour):
        hour = float(hour)
        if hour not in self.departs:
            self.departs[hour] = self._build(hour, forward=True)
        return self.departs[hour]

    def clear(self):
        self.views.clear()
        self.departs.clear()

    def round_trip(self, dep):
        return dep["out"] + dep["home"]

    def accepts(self, dep, dwell_Wh=0.0):
        return self.round_trip(dep) + dwell_Wh <= budget_Wh()


def cell(idx, shape):
    return divmod(int(idx), shape[1])


def outbound_path(dep, target):
    path = [int(target)]
    pr = dep["out_pred"]
    while pr[path[-1]] != NO_PRED:
        path.append(int(pr[path[-1]]))
        if len(path) > pr.size:
            raise RuntimeError("loop in outbound predecessors")
    return path[::-1]


def home_path(view, start):
    path = [int(start)]
    pr = view["home_pred"]
    while pr[path[-1]] != NO_PRED:
        path.append(int(pr[path[-1]]))
        if len(path) > pr.size:
            raise RuntimeError("loop in home predecessors")
    return path


def step_index(a, b, shape):
    (r0, c0), (r1, c1) = cell(a, shape), cell(b, shape)
    return STEPS.index((r1 - r0, c1 - c0))


def path_cost(path, costs, shape):
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        r, c = cell(a, shape)
        total += costs[step_index(a, b, shape), r, c] + 1e-9
    return total