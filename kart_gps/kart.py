"""2D kart physics: four tyres on a flat track, solid rear axle, steered front wheels.

The kart moves in the ground plane (x, y, heading). Vertical effects are worked out
from formulas each step instead of being simulated as motion:
  - load on each tyre: static 45/55 split + weight transfer from braking/accelerating
    and cornering (centre of mass 0.2 m up), plus steering "jacking" that unloads the
    inside rear tyre the way caster and frame flex do on a real kart
  - each tyre's grip depends on its load (simplified Pacejka with combined slip:
    a tyre that is braking hard has less grip left for cornering)
  - rollover: lateral acceleration above tip_lateral_g

Driveline: one rear axle (both rear wheels turn at the same speed, so the inside rear
tyre scrubs in corners, like a real kart). Motor force/regen and the hydraulic brake
act on that axle. Front wheels roll freely and are steered with Ackermann geometry.

Inputs each physics step (set by control.LowLevel): motor force (N at the tyres,
+ drive / - regen), brake command (0..1 of full force) and steering angle (rad, the
bicycle-model angle at the axle centre, + = left). The actuators then add their own
limits and lags: motor lag, top-speed taper, regen power limit, brake actuator travel,
steering delay and rate limit.

Frame: world x = east, y = north. Body x forward, y left. State is at the centre of mass.
"""
import math
from collections import deque

G = 9.81
AIR = 1.2
V_LOW = 1.0          # m/s, slip ratios/angles use at least this speed (keeps standstill stable)


class Kart:
    def __init__(self, p, dt):
        """p: kart parameter dict (gps.yaml 'kart', maybe randomised). dt: physics step (s)."""
        self.p = p
        self.dt = dt
        self.m = p["mass"]
        self.L = p["wheelbase"]
        # Centre of mass: front_weight of the load on the front axle -> CG is
        # front_weight * L ahead of the rear axle.
        self.lr = p["front_weight"] * self.L          # rear axle -> CG
        self.lf = self.L - self.lr                    # CG -> front axle
        self.h = p["cg_height"]
        self.Iz = p["yaw_inertia"]
        self.Rf = p["tyre_front"]["diameter"] / 2
        self.Rr = p["tyre_rear"]["diameter"] / 2
        self.tf, self.tr = p["track_front"], p["track_rear"]
        self.kh = p["kingpin_half"]
        self.Iw = p["axle_inertia"]
        # wheel positions from the CG: FL, FR, RL, RR
        self.wx = (self.lf, self.lf, -self.lr, -self.lr)
        self.wy = (self.tf / 2, -self.tf / 2, self.tr / 2, -self.tr / 2)
        # static loads
        wf = p["front_weight"]
        self.fz0 = (self.m * G * wf / 2,) * 2 + (self.m * G * (1 - wf) / 2,) * 2
        self.mu = (p["mu"] * p["front_grip"],) * 2 + (p["mu"],) * 2
        C = p["shape"]
        self.C = C
        self.B = math.tan(math.pi / (2 * C))     # peak at normalised slip = 1
        self.kp = p["slip_ratio_peak"]
        self.tap = math.tan(math.radians(p["slip_angle_peak_deg"]))
        # steering: max bicycle angle so the inner wheel stays within max_steer_deg
        t_in = math.tan(math.radians(p["max_steer_deg"]))
        self.max_delta = math.atan(t_in * self.L / (self.L + self.kh * t_in))
        self.steer_rate = math.radians(p["steer_rate_deg"])
        self.steer_buf = deque([0.0] * max(1, int(round(p["steer_delay"] / dt))))
        self.reset(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ state
    def reset(self, x, y, yaw, speed=0.0):
        """Place the kart with its REAR AXLE CENTRE at (x, y), heading yaw, rolling at speed."""
        self.yaw = yaw
        self.x = x + self.lr * math.cos(yaw)
        self.y = y + self.lr * math.sin(yaw)
        self.vx, self.vy, self.r = speed, 0.0, 0.0
        self.omega = speed / self.Rr              # rear axle, rad/s
        self.delta = 0.0                          # actual bicycle steering angle
        self.f_motor = 0.0                        # N at the tyres (+ drive, - regen)
        held = self.p.get("brake_hold_at_bite", True)
        self.brake_pos = 0.0 if held else -self.p["brake_dead_time"]   # s of actuator travel
        self.ax = self.ay = 0.0                   # what an accelerometer at the CG reads
        self.fz = list(self.fz0)
        self.fx = [0.0] * 4
        self.fy = [0.0] * 4
        self.steer_buf = deque([0.0] * len(self.steer_buf))
        self.rolled = False
        self.t = 0.0

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    @property
    def rear_axle(self):
        """World position of the rear axle centre (the GNSS mast is straight above it)."""
        return (self.x - self.lr * math.cos(self.yaw), self.y - self.lr * math.sin(self.yaw))

    @property
    def axle_speed(self):
        """Rear wheel surface speed (m/s), what the wheel encoder measures."""
        return self.omega * self.Rr

    @property
    def brake_force_now(self):
        return max(0.0, min(1.0, self.brake_pos / self.p["brake_apply_time"])) * self.p["brake_force"]

    def wheel_angles(self, delta=None):
        """Ackermann: (left, right) wheel angles for a bicycle angle delta."""
        t = math.tan(self.delta if delta is None else delta)
        L, kh = self.L, self.kh
        return math.atan(L * t / (L - kh * t)), math.atan(L * t / (L + kh * t))

    def tyre_corners(self):
        """Corners of the four tyres seen from above, world frame: list of 16 (x, y).
        A tyre is a rectangle (diameter long, width wide), turned with its steering angle."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dl, dr = self.wheel_angles()
        out = []
        for i, ang in enumerate((dl, dr, 0.0, 0.0)):
            tyre = self.p["tyre_front"] if i < 2 else self.p["tyre_rear"]
            hl, hw = tyre["diameter"] / 2, tyre["width"] / 2
            ca, sa = math.cos(ang), math.sin(ang)
            for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
                bx = self.wx[i] + a * ca - b * sa
                by = self.wy[i] + a * sa + b * ca
                out.append((self.x + bx * c - by * s, self.y + bx * s + by * c))
        return out

    # ------------------------------------------------------------------ tyres
    def _tyre(self, i, kappa, tan_a):
        """(Fx, Fy) in the wheel frame for slip ratio kappa and tan(slip angle)."""
        fz = self.fz[i]
        if fz <= 0.0:
            return 0.0, 0.0
        sx, sy = kappa / self.kp, tan_a / self.tap
        s = math.hypot(sx, sy)
        if s < 1e-9:
            return 0.0, 0.0
        mu = self.mu[i] * (1.0 - self.p["load_sensitivity"] * (fz / self.fz0[i] - 1.0))
        f = mu * fz * math.sin(self.C * math.atan(self.B * s))
        return f * sx / s, f * sy / s

    def _rear_fx(self, omega, vxw):
        """Total longitudinal force of the two rear tyres for axle speed omega."""
        tot = 0.0
        for i in (2, 3):
            kappa = (omega * self.Rr - vxw[i]) / max(abs(vxw[i]), V_LOW)
            tot += self._tyre(i, kappa, self._tan_a[i])[0]
        return tot

    # ------------------------------------------------------------------ step
    def step(self, motor_cmd, brake_cmd, steer_cmd):
        """Advance one physics step. motor_cmd N (+ drive / - regen), brake_cmd 0..1,
        steer_cmd rad (bicycle angle)."""
        p, dt, m = self.p, self.dt, self.m

        # --- actuators
        self.steer_buf.append(max(-self.max_delta, min(self.max_delta, steer_cmd)))
        target = self.steer_buf.popleft()
        step = self.steer_rate * dt
        self.delta += max(-step, min(step, target - self.delta))

        v_wheel = self.omega * self.Rr
        top, taper = p["top_speed"], p["top_speed_taper"]
        f_drive_max = p["drive_force"] * max(0.0, min(1.0, (top - v_wheel) / taper))
        # regen: battery charge-power limit, fades out near standstill
        f_regen_max = min(p["regen_force"], p["regen_power"] / max(v_wheel, 0.1)) * min(1.0, v_wheel / 1.0)
        f_cmd = max(-f_regen_max, min(f_drive_max, motor_cmd))
        self.f_motor += (f_cmd - self.f_motor) * min(1.0, dt / p["motor_lag"])

        held = p.get("brake_hold_at_bite", True)
        lo = 0.0 if held else -p["brake_dead_time"]
        b_target = lo if brake_cmd <= 0.0 else min(1.0, brake_cmd) * p["brake_apply_time"]
        self.brake_pos += max(-dt, min(dt, b_target - self.brake_pos))   # 1 s of travel per s
        f_brake = self.brake_force_now

        # --- tyre loads (weight transfer from last step's accelerations)
        L, h = self.L, self.h
        wf = p["front_weight"]
        fz_front = m * G * wf - m * self.ax * h / L
        fz_rear = m * G * (1 - wf) + m * self.ax * h / L
        rs = p["rear_roll_share"]
        dfz_f = m * self.ay * h * (1 - rs) / self.tf
        dfz_r = m * self.ay * h * rs / self.tr
        # Steering jacking: caster pushes the inside front wheel down, which twists the
        # frame and unloads the inside rear (+ outside front). This is what lets a kart
        # with a solid rear axle turn. + = turning left, inside = left.
        warp = p["steer_jacking"] * self.delta
        # + ay (turning left) moves load to the right-hand tyres
        self.fz = [max(0.0, fz_front / 2 - dfz_f + warp), max(0.0, fz_front / 2 + dfz_f - warp),
                   max(0.0, fz_rear / 2 - dfz_r - warp), max(0.0, fz_rear / 2 + dfz_r + warp)]
        if abs(self.ay) > p["tip_lateral_g"] * G:
            self.rolled = True

        # --- wheel-frame velocities and slip angles
        dl, dr = self.wheel_angles()
        angles = (dl, dr, 0.0, 0.0)
        vxw, vyw, self._tan_a = [0.0] * 4, [0.0] * 4, [0.0] * 4
        for i in range(4):
            vxi = self.vx - self.r * self.wy[i]
            vyi = self.vy + self.r * self.wx[i]
            ca, sa = math.cos(angles[i]), math.sin(angles[i])
            vxw[i] = vxi * ca + vyi * sa
            vyw[i] = -vxi * sa + vyi * ca
            self._tan_a[i] = -vyw[i] / max(abs(vxw[i]), V_LOW)

        # --- rear axle, implicit in the tyre force (stiff at low speed)
        t_motor = self.f_motor * self.Rr
        t_brake = f_brake * self.Rr
        eps = 0.01
        f0 = self._rear_fx(self.omega, vxw)
        dfdw = (self._rear_fx(self.omega + eps, vxw) - f0) / eps
        i_eff = self.Iw + dt * self.Rr * max(0.0, dfdw)
        om = self.omega + dt * (t_motor - t_brake - self.Rr * f0) / i_eff
        if om < 0.0:        # brake/regen can stop the axle but not spin it backwards
            om = 0.0
        self.omega = om

        # --- tyre forces -> body frame
        fx_b = fy_b = mz = 0.0
        for i in range(4):
            if i >= 2:
                kappa = (self.omega * self.Rr - vxw[i]) / max(abs(vxw[i]), V_LOW)
            else:
                kappa = 0.0
            fxw, fyw = self._tyre(i, kappa, self._tan_a[i])
            if i < 2:   # free-rolling front: rolling resistance only
                fxw = -p["rolling_resistance"] * self.fz[i] * _sgn(vxw[i])
            ca, sa = math.cos(angles[i]), math.sin(angles[i])
            fxi = fxw * ca - fyw * sa
            fyi = fxw * sa + fyw * ca
            self.fx[i], self.fy[i] = fxi, fyi
            fx_b += fxi
            fy_b += fyi
            mz += self.wx[i] * fyi - self.wy[i] * fxi
        # rear rolling resistance + air drag (never push the kart backwards)
        resist = p["rolling_resistance"] * (self.fz[2] + self.fz[3]) + 0.5 * AIR * p["drag_area"] * self.vx ** 2
        fx_b -= _sgn(self.vx) * min(resist, m * abs(self.vx) / dt)

        # --- body dynamics (semi-implicit Euler)
        self.ax, self.ay = fx_b / m, fy_b / m
        self.vx += (self.ax + self.vy * self.r) * dt
        self.vy += (self.ay - self.vx * self.r) * dt
        self.r += mz / self.Iz * dt
        if self.vx < 0.0 and self.omega == 0.0:
            self.vx = 0.0
        # Blend to the kinematic model near standstill, where slip makes no sense.
        w = max(0.0, min(1.0, (self.vx - 0.5) / 1.5))
        if w < 1.0:
            r_kin = self.vx * math.tan(self.delta) / L
            self.r = w * self.r + (1 - w) * r_kin
            self.vy = w * self.vy + (1 - w) * r_kin * self.lr
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.x += (self.vx * c - self.vy * s) * dt
        self.y += (self.vx * s + self.vy * c) * dt
        self.yaw = _wrap(self.yaw + self.r * dt)
        self.t += dt


def _sgn(v):
    return 1.0 if v > 0 else (-1.0 if v < 0 else 0.0)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi
