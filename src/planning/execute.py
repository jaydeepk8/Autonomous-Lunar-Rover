# moving between sorties

import sys
sys.path.insert(0, "src/illumination")
sys.path.insert(0, "src/energy")
sys.path.insert(0, "src/planning")

import math
import numpy as np

from sun_geometry import sun_position, SYNODIC_HOURS
from horizon_field import R_MOON, DISC_RADIUS_DEG
from irradiance import PARAMS as SUN_PARAMS
from loads import p as load_p, drive_power
from battery import p as batt_p
from coupling import PARAMS as FAULT
from edges import STEPS, p as edge_p
from planner import budget_Wh, charge_power, outbound_path, NO_PRED

PARAMS = {
    "stop_step_h": dict(
        value=0.25, unit="h", tag="assumed",
        source="time step while stopped, so the sun keeps moving during a fault"),
    "max_hours": dict(
        value=240.0, unit="h", tag="assumed",
        source="a sortie still out after this long counts as stranded"),
    "dwell_h": dict(
        value=0.0, unit="h", tag="assumed",
        source="time parked at the target; zero matches the Phase 2 round trip"),
}

STEP_K = {s: k for k, s in enumerate(STEPS)}
SUN_TABLE_H = 1.0 / 60.0


def p(name):
    return PARAMS[name]["value"]


class State:
    __slots__ = ("t", "idx", "soc", "soc0", "min_soc", "ledger", "faults",
                 "hazard", "next_fault", "rng", "trace", "reroutes", "waits")


class World:
    """The truth both planners face: moving sun, random faults, battery tracked per step."""

    def __init__(self, terr, H, q, rate="2023", frozen=None):
        self.terr, self.H, self.q, self.rate = terr, H, q, rate
        self.shape = terr["passable"].shape
        self.passable = terr["passable"]
        self.xs = [float(x) for x in terr["xs"]]
        self.ys = [float(y) for y in terr["ys"]]
        self.L = np.stack([s["L"] for s in terr["steps"]])
        self.P_drive = np.stack([drive_power(s["grade"]) for s in terr["steps"]])
        self.step_az = [float(s["az"]) for s in terr["steps"]]
        self.lam0 = FAULT[f"lambda0_{rate}"]["value"]
        self.T_rec = FAULT[f"T_rec_{rate}"]["value"] / 3600.0
        self.peak = SUN_PARAMS["P_array_peak"]["value"]
        self.v = load_p("v_drive")
        self.cap = batt_p("E_batt")
        self.floor = self.cap * (1.0 - batt_p("dod"))
        self.eta = batt_p("eta_charge")

        self.frozen = None if frozen is None else float(frozen)
        if self.frozen is None:
            hrs = np.arange(0.0, SYNODIC_HOURS, SUN_TABLE_H)
            el, az = sun_position(hrs)
            self.sun_el, self.sun_az = el.tolist(), az.tolist()
        else:
            el, az = sun_position(self.frozen)
            self.sun_el, self.sun_az = [float(el)], [float(az)]

    def plan_time(self, t):
        return t if self.frozen is None else self.frozen

    def sun(self, t):
        if self.frozen is not None:
            return self.sun_el[0], self.sun_az[0]
        i = int(round(t / SUN_TABLE_H)) % len(self.sun_el)
        return self.sun_el[i], self.sun_az[i]

    def cell_sun(self, r, c, t):
        el, az = self.sun(t)
        a = math.radians(az)
        e = el + math.degrees((self.xs[c] * math.sin(a) - self.ys[r] * math.cos(a)) / R_MOON)
        n_az = self.H.shape[2]
        x = (az % 360.0) / 360.0 * n_az
        k0 = int(math.floor(x)) % n_az
        w = x - math.floor(x)
        # float32 like horizon_field.horizon_at, so lit/dark agrees with the planner
        h = np.float32(1.0 - w) * self.H[r, c, k0] + np.float32(w) * self.H[r, c, (k0 + 1) % n_az]
        f = min(max((e + DISC_RADIUS_DEG - float(h)) / (2.0 * DISC_RADIUS_DEG), 0.0), 1.0)
        return f > 0.0, self.peak * f * max(math.cos(math.radians(e)), 0.0), az

    def idle(self, lit):
        return load_p("P_avionics") + (load_p("P_heater_sun") if lit else load_p("P_heater_shadow"))

    def spend(self, st, load, P_sun, dt):
        net = P_sun - load
        st.soc = min(st.soc + (net * self.eta if net > 0.0 else net) * dt, self.cap)
        st.min_soc = min(st.min_soc, st.soc)
        # planner-style tally: no credit for surplus sunlight
        st.ledger += max(load - P_sun, 0.0) * dt
        st.t += dt

    def q_now(self, lit, lights, k, az):
        q = self.q
        if not lit:
            return q["headlight"] if lights else q["dark"]
        phase = 180.0 - abs((self.step_az[k] - az + 180.0) % 360.0 - 180.0)
        w = min(max((phase - edge_p("phase_low")) / (edge_p("phase_high") - edge_p("phase_low")),
                    0.0), 1.0)
        return q["sunlit_low"] * (1.0 - w) + q["sunlit_high"] * w

    def move(self, st, b, kappa, lights_rule):
        r, c = divmod(st.idx, self.shape[1])
        r1, c1 = divmod(b, self.shape[1])
        k = STEP_K[(r1 - r, c1 - c)]
        lit, P_sun, az = self.cell_sun(r, c, st.t)
        lights = (not lit) and lights_rule
        L = float(self.L[k, r, c])
        load = self.idle(lit) + float(self.P_drive[k, r, c]) + (load_p("P_light") if lights else 0.0)
        self.spend(st, load, P_sun, L / self.v / 3600.0)
        st.idx = b

        q = self.q_now(lit, lights, k, az)
        st.hazard += self.lam0 * (1.0 + kappa * min(max(1.0 - q, 0.0), 1.0)) * L
        n = 0
        while st.hazard >= st.next_fault:
            n += 1
            st.next_fault += st.rng.exponential()
        if st.trace is not None:
            st.trace.append((st.t, b, st.soc, lit, lights, ""))
        return n

    def stop(self, st, hours, event=""):
        left = hours
        r, c = divmod(st.idx, self.shape[1])
        while left > 1e-12:
            dt = min(p("stop_step_h"), left)
            lit, P_sun, _ = self.cell_sun(r, c, st.t)
            self.spend(st, self.idle(lit), P_sun, dt)
            left -= dt
            if st.soc <= self.floor:
                return False
        if st.trace is not None:
            st.trace.append((st.t, st.idx, st.soc, lit, False, event))
        return True

    def charging(self, idx, t):
        r, c = divmod(idx, self.shape[1])
        return bool(self.passable[r, c]) and self.cell_sun(r, c, t)[1] >= charge_power()


def fly(world, planner, dep, target, kappa_true, rng=None, forced_out=(),
        forced_target=0, turn_back=True, record=False):
    path = outbound_path(dep, target)
    dwell_Wh = p("dwell_h") * world.idle(False)
    st = State()
    st.t, st.idx = float(dep["hour"]), path[0]
    st.soc = st.soc0 = st.min_soc = world.cap
    st.ledger, st.faults, st.hazard = 0.0, 0, 0.0
    st.reroutes, st.waits = 0, 0
    st.rng = rng
    st.next_fault = rng.exponential() if rng is not None else math.inf
    st.trace = [(st.t, st.idx, st.soc, True, False, "depart")] if record else None
    t0, allow = st.t, st.soc0 - budget_Wh()

    leg, i, reached, turned, reason = "out", 0, False, False, None
    decisions = []
    home_view = None
    bucket = planner.bucket(world.plan_time(st.t))

    def fault(event):
        st.faults += 1
        return world.stop(st, world.T_rec, event)

    while reason is None:
        if st.t - t0 > p("max_hours"):
            reason = "timeout"
            break

        if leg == "out" and i == len(path) - 1:
            reached = True
            if st.trace is not None:
                st.trace.append((st.t, st.idx, st.soc, False, False, "target"))
            if dwell_Wh > 0.0 and not world.stop(st, p("dwell_h"), "dwell"):
                reason = "battery floor"
                break
            for _ in range(forced_target):
                if not fault("forced fault"):
                    reason = "battery floor"
                    break
            leg = "home"
            continue

        if leg == "out":
            nxt = path[i + 1]
        else:
            if world.charging(st.idx, st.t):
                break
            v = planner.view(world.plan_time(st.t))
            # never fall back to an older plan than the one already in use
            if home_view is None or v["hour"] > home_view["hour"]:
                home_view = v
            if home_view["home"][st.idx] == 0.0:
                # planned charging cell has lost the sun: replan now, not at the next refresh
                step = p("stop_step_h")
                home_view = planner.view_at(world.plan_time(math.ceil(st.t / step) * step))
                st.reroutes += 1
            view = home_view
            nxt = int(view["home_pred"][st.idx])
            if nxt == NO_PRED:
                # no charging cell reachable right now: wait for the sun
                st.waits += 1
                if not world.stop(st, p("stop_step_h"), "wait"):
                    reason = "battery floor"
                continue

        n = world.move(st, nxt, kappa_true, planner.lights)
        if leg == "out":
            i += 1
            n += int(i in forced_out)
        if st.soc <= world.floor:
            reason = "battery floor"
            break
        for _ in range(n):
            if not fault("fault"):
                reason = "battery floor"
                break
        if reason is not None:
            break

        if leg == "out" and turn_back and i < len(path) - 1:
            b = planner.bucket(world.plan_time(st.t))
            if n or b != bucket:
                bucket = b
                view = planner.view(world.plan_time(st.t))
                need = (dep["out"][target] - dep["out"][st.idx]) + float(view["home"][target]) + dwell_Wh
                go_back = st.soc - need < allow
                decisions.append(dict(t=st.t, move=i, soc=st.soc, need=need,
                                      allowance=st.soc - allow, turn_back=go_back))
                if go_back:
                    leg, turned = "home", True
                    if st.trace is not None:
                        st.trace.append((st.t, st.idx, st.soc, False, False, "turn back"))

    if st.trace is not None:
        st.trace.append((st.t, st.idx, st.soc, True, False, reason or "home"))
    drawdown = st.soc0 - st.min_soc
    breach = drawdown > budget_Wh()
    if reason is not None:
        outcome = "stranded"
    elif breach:
        outcome = "over budget"
    elif turned:
        outcome = "turned back"
    else:
        outcome = "within budget"
    return dict(outcome=outcome, reason=reason, reached=reached, turned_back=turned,
                breach=breach, drawdown=drawdown, ledger=st.ledger, faults=st.faults,
                expected_faults=st.hazard, hours=st.t - t0,
                reroutes=st.reroutes, waits=st.waits,
                predicted=float(dep["out"][target] + dep["home"][target]),
                decisions=decisions, trace=st.trace)