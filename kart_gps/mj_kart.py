"""MuJoCo kart: aks_kart.xml on the surveyed track, with the same interface as kart.Kart,
so LapEnv, the sensors, the EKF, the low-level controller and the drivers run unchanged.

    cfg["physics"] = "mujoco"            # LapEnv then builds an MjKart instead of a Kart

What MuJoCo simulates (instead of the 2D formulas): the chassis as a rigid 3D body on
four tyre contacts, so weight transfer, the solid rear axle scrubbing, roll and a real
rollover come out of the contact physics. Tyre grip is MuJoCo's contact friction
(Coulomb, no slip curve), set from the kart's mu and front_grip times the gps.yaml
'mujoco' scales, which are tuned so scripts/gps/kart_checks.py --mujoco matches the
2D kart.

What is copied from kart.Kart (the actuators, not physics): steering delay, rate limit
and Ackermann (the servos then follow the wheel angles), motor lag, top-speed taper,
regen power limit, brake actuator travel. Rolling resistance and air drag are a force
on the chassis.

The scene is built in memory from tracks/.../track_world.xml with its line_robot.xml
include swapped for aks_kart.xml; neither file is changed. One model + data per process
(and per scene options) is reused by every MjKart, so a viewer can hold on to it.
"""
import math
import pathlib
import re
import xml.etree.ElementTree as ET
from collections import deque

import mujoco
import numpy as np

from .config import ROOT, track_path

G = 9.81
AIR = 1.2
KART_XML = pathlib.Path(__file__).with_name("aks_kart.xml")
TYRES = ("front_left_wheel_geom", "front_right_wheel_geom", "rear_left_wheel_geom", "rear_right_wheel_geom")
Z0 = 0.1395            # rear axle height = rear tyre radius (body origin in aks_kart.xml)

_SCENES = {}


def _no_comments(path):
    """File text without <!-- --> comments (aks_kart.xml's has a '--', which MuJoCo
    accepts but strict XML parsers don't)."""
    return re.sub(r"<!--.*?-->", "", pathlib.Path(path).read_text(encoding="utf-8"), flags=re.S)


def _match_cg(kart, h):
    """Move the ballast block up or down so the kart's centre of mass is h above the
    ground, like the 2D kart's cg_height (the XML's parts alone sit lower)."""
    ballast = kart.find(".//geom[@name='body_and_ballast']")
    m = mujoco.MjModel.from_xml_string(ET.tostring(kart, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rob = m.body("robot").id
    z_now = d.subtree_com[rob][2]
    pos = [float(v) for v in ballast.get("pos").split()]
    pos[2] += (h - z_now) * m.body_subtreemass[rob] / float(ballast.get("mass"))
    ballast.set("pos", " ".join(f"{v:.4f}" for v in pos))


def scene_xml(cfg, lines=False):
    """Track world + aks_kart.xml as one MJCF string. lines: keep the 1732 edge markers
    (drawing only, slows collision checks a little)."""
    track = ET.fromstring(_no_comments(track_path(cfg)))
    kart = ET.fromstring(_no_comments(KART_XML))
    for inc in track.findall("include"):
        track.remove(inc)
    wb = track.find("worldbody")
    if not lines:
        for g in [g for g in wb.findall("geom") if g.get("name", "").startswith("line")]:
            wb.remove(g)
    for tag in ("asset", "default"):
        dst = track.find(tag)
        if dst is None:
            dst = ET.SubElement(track, tag)
        for child in kart.find(tag):
            dst.append(child)
    _match_cg(kart, cfg["kart"]["cg_height"])
    for body in kart.find("worldbody"):
        wb.append(body)
    for tag in ("actuator", "sensor"):
        track.append(kart.find(tag))
    vis = track.find("visual")
    if vis is not None:            # the track file's tiny near/far planes clip a close-up kart
        for m in vis.findall("map"):
            vis.remove(m)
        ET.SubElement(vis, "map", znear="0.01", zfar="500")
        for g in vis.findall("global"):
            g.set("offwidth", "1920")
            g.set("offheight", "1080")
    return ET.tostring(track, encoding="unicode")


def scene(cfg, lines=False):
    """(model, data) for this process, built once per scene option set."""
    mc = cfg["mujoco"]
    key = (str(track_path(cfg)), lines, mc.get("cone", "pyramidal"), mc.get("impratio", 1.0))
    if key not in _SCENES:
        m = mujoco.MjModel.from_xml_string(scene_xml(cfg, lines))
        m.opt.cone = (mujoco.mjtCone.mjCONE_ELLIPTIC if mc.get("cone") == "elliptic"
                      else mujoco.mjtCone.mjCONE_PYRAMIDAL)
        m.opt.impratio = float(mc.get("impratio", 1.0))
        # rolling resistance is a force in MjKart.step, so no drag in the wheel/axle joints
        for j in ("front_left_spin", "front_right_spin", "rear_axle_joint"):
            m.dof_damping[m.joint(j).dofadr[0]] = 0.0
        nominal = {"mass": m.body_mass.copy(), "inertia": m.body_inertia.copy()}
        _SCENES[key] = (m, mujoco.MjData(m), nominal)
    return _SCENES[key]


class MjKart:
    def __init__(self, p, dt, cfg, lines=False):
        """p: kart parameter dict (maybe randomised), dt: physics step, cfg: full config."""
        self.p, self.dt = p, dt
        self.mc = mc = cfg["mujoco"]
        self.m, self.d, nom = scene(cfg, lines)
        m = self.m
        self.substeps = int(mc.get("substeps", 1))
        m.opt.timestep = dt / self.substeps
        self.L, self.kh = p["wheelbase"], p["kingpin_half"]
        self.Rf = p["tyre_front"]["diameter"] / 2
        self.Rr = p["tyre_rear"]["diameter"] / 2

        # ids
        self.body = m.body("robot").id
        self.q_free = m.joint("robot_free").qposadr[0]
        self.v_free = m.joint("robot_free").dofadr[0]
        self.v_axle = m.joint("rear_axle_joint").dofadr[0]
        self.q_steer = [m.joint(n).qposadr[0] for n in ("left_steer_joint", "right_steer_joint")]
        self.v_front = [m.joint(n).dofadr[0] for n in ("front_left_spin", "front_right_spin")]
        self.a_axle, self.a_sl, self.a_sr = (m.actuator(n).id for n in ("axle_torque", "steer_left", "steer_right"))
        self.g_tyre = [m.geom(n).id for n in TYRES]
        self.s_acc = m.sensor("imu_accel").adr[0]
        self.s_gyro = m.sensor("imu_gyro").adr[0]
        self.site_gnss = m.site("gnss").id

        # this kart's numbers: mass, grip, contact stiffness
        f = p["mass"] / 125.0
        m.body_mass[:] = nom["mass"]
        m.body_inertia[:] = nom["inertia"]
        m.body_mass[self.body] *= f
        m.body_inertia[self.body] *= f
        mu_r = p["mu"] * mc["friction_scale"]
        mu_f = mu_r * p["front_grip"] * mc["front_friction_scale"]
        for i, g in enumerate(self.g_tyre):
            m.geom_friction[g, 0] = mu_f if i < 2 else mu_r
            m.geom_solref[g] = mc["solref"]
            m.geom_condim[g] = int(mc.get("condim", 6))
            if "solimp" in mc:
                m.geom_solimp[g, :len(mc["solimp"])] = mc["solimp"]
        # rear axle + wheels + motor through the gearing = the 2D kart's axle_inertia
        m.dof_armature[self.v_axle] = 0.0
        mujoco.mj_forward(m, self.d)
        M = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, self.d, M)
        own = float(M[self.v_axle, self.v_axle])
        m.dof_armature[self.v_axle] = max(0.0, p["axle_inertia"] - own)
        mujoco.mj_setConst(m, self.d)
        self.mass = float(m.body_subtreemass[self.body])

        t_in = math.tan(math.radians(p["max_steer_deg"]))
        self.max_delta = math.atan(t_in * self.L / (self.L + self.kh * t_in))
        self.steer_rate = math.radians(p["steer_rate_deg"])
        self.steer_buf = deque([0.0] * max(1, int(round(p["steer_delay"] / dt))))
        self.acc_lp = math.exp(-2 * math.pi * mc["imu_filter_hz"] * dt)
        self.reset(0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ state
    def reset(self, x, y, yaw, speed=0.0):
        """Rear axle centre at (x, y), heading yaw, rolling at speed (as Kart.reset)."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        q = self.q_free
        d.qpos[q:q + 3] = (x, y, Z0)
        d.qpos[q + 3:q + 7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
        for _ in range(int(0.3 / m.opt.timestep)):     # settle onto the tyres, standing still
            mujoco.mj_step(m, d)
        d.qpos[q:q + 2] = (x, y)
        c, s = math.cos(yaw), math.sin(yaw)
        d.qvel[self.v_free:self.v_free + 3] = (speed * c, speed * s, 0.0)
        d.qvel[self.v_free + 3:self.v_free + 6] = 0.0
        d.qvel[self.v_axle] = speed / self.Rr
        for v in self.v_front:
            d.qvel[v] = speed / self.Rf
        d.time = 0.0
        mujoco.mj_forward(m, d)
        self.delta = 0.0
        self.f_motor = 0.0
        held = self.p.get("brake_hold_at_bite", True)
        self.brake_pos = 0.0 if held else -self.p["brake_dead_time"]
        self.steer_buf = deque([0.0] * len(self.steer_buf))
        self.rolled = False
        self.t = 0.0
        self.axle_torque = 0.0
        self._read(first=True)

    def _read(self, first=False):
        """Pull the kart state out of MuJoCo (same names as Kart)."""
        m, d = self.m, self.d
        R = d.xmat[self.body]
        self.yaw = math.atan2(R[3], R[0])
        self.up = R[8]                                  # body z . world z (1 = flat)
        # free joint: linear velocity of the body origin (rear axle centre) in the world
        # frame, angular velocity in the body frame
        vo = d.qvel[self.v_free:self.v_free + 3]
        w = d.qvel[self.v_free + 3:self.v_free + 6]
        wz = R[6] * w[0] + R[7] * w[1] + R[8] * w[2]   # yaw rate (world z)
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        vrx, vry = vo[0] * c + vo[1] * s, -vo[0] * s + vo[1] * c
        self.r = wz
        com = d.subtree_com[self.body] - d.xpos[self.body]
        self.lr = com[0] * c + com[1] * s               # rear axle -> centre of mass
        self.vx, self.vy = vrx, vry + wz * self.lr
        self.x, self.y = float(d.subtree_com[self.body][0]), float(d.subtree_com[self.body][1])
        self.omega = float(d.qvel[self.v_axle])
        # accelerometer: specific force minus gravity seen through the tilt, low-passed
        a = d.sensordata[self.s_acc:self.s_acc + 3]
        if first:
            self.ax = self.ay = 0.0
        else:
            k = self.acc_lp
            self.ax = k * self.ax + (1 - k) * float(a[0])
            self.ay = k * self.ay + (1 - k) * float(a[1])
        self.gyro_z = float(d.sensordata[self.s_gyro + 2])
        if self.up < math.cos(math.radians(35)):
            self.rolled = True

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)

    @property
    def rear_axle(self):
        p = self.d.xpos[self.body]
        return (float(p[0]), float(p[1]))

    @property
    def antenna(self):
        """Where the GNSS antenna really is (moves sideways when the kart rolls)."""
        p = self.d.site_xpos[self.site_gnss]
        return (float(p[0]), float(p[1]))

    @property
    def axle_speed(self):
        return self.omega * self.Rr

    @property
    def brake_force_now(self):
        return max(0.0, min(1.0, self.brake_pos / self.p["brake_apply_time"])) * self.p["brake_force"]

    def wheel_angles(self, delta=None):
        t = math.tan(self.delta if delta is None else delta)
        L, kh = self.L, self.kh
        return math.atan(L * t / (L - kh * t)), math.atan(L * t / (L + kh * t))

    def tyre_corners(self):
        """Corners of the four tyres seen from above (FL FR RL RR), from MuJoCo's tyre
        positions and the actual steering joint angles."""
        out = []
        d = self.d
        for i, g in enumerate(self.g_tyre):
            tyre = self.p["tyre_front"] if i < 2 else self.p["tyre_rear"]
            hl, hw = tyre["diameter"] / 2, tyre["width"] / 2
            ang = self.yaw + (float(d.qpos[self.q_steer[i]]) if i < 2 else 0.0)
            ca, sa = math.cos(ang), math.sin(ang)
            cx, cy = d.geom_xpos[g][0], d.geom_xpos[g][1]
            for a, b in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)):
                out.append((cx + a * ca - b * sa, cy + a * sa + b * ca))
        return out

    # ------------------------------------------------------------------ step
    def step(self, motor_cmd, brake_cmd, steer_cmd):
        """One physics step, same inputs as Kart.step."""
        p, dt, d = self.p, self.dt, self.d

        # steering: delay, rate limit, Ackermann -> the two servos
        self.steer_buf.append(max(-self.max_delta, min(self.max_delta, steer_cmd)))
        target = self.steer_buf.popleft()
        st = self.steer_rate * dt
        self.delta += max(-st, min(st, target - self.delta))
        dl, dr = self.wheel_angles()
        d.ctrl[self.a_sl], d.ctrl[self.a_sr] = dl, dr

        # motor / regen / brake -> one rear-axle torque
        v_wheel = max(0.0, self.omega * self.Rr)
        top, taper = p["top_speed"], p["top_speed_taper"]
        f_drive_max = p["drive_force"] * max(0.0, min(1.0, (top - v_wheel) / taper))
        f_regen_max = min(p["regen_force"], p["regen_power"] / max(v_wheel, 0.1)) * min(1.0, v_wheel / 1.0)
        f_cmd = max(-f_regen_max, min(f_drive_max, motor_cmd))
        self.f_motor += (f_cmd - self.f_motor) * min(1.0, dt / p["motor_lag"])
        held = p.get("brake_hold_at_bite", True)
        lo = 0.0 if held else -p["brake_dead_time"]
        b_target = lo if brake_cmd <= 0.0 else min(1.0, brake_cmd) * p["brake_apply_time"]
        self.brake_pos += max(-dt, min(dt, b_target - self.brake_pos))
        t_brake = self.brake_force_now * self.Rr
        # the brake opposes the axle's turning; faded near standstill so it holds without chatter
        tq = self.f_motor * self.Rr - t_brake * max(-1.0, min(1.0, self.omega / 0.5))
        self.axle_torque = tq
        d.ctrl[self.a_axle] = tq

        # rolling resistance + air drag on the chassis, against its ground velocity
        vg = d.qvel[self.v_free:self.v_free + 2]
        v = math.hypot(vg[0], vg[1])
        f = p["rolling_resistance"] * self.mass * G * min(1.0, v / 0.1) + 0.5 * AIR * p["drag_area"] * v * v
        d.xfrc_applied[self.body, :] = 0.0
        if v > 1e-6:
            d.xfrc_applied[self.body, 0] = -f * vg[0] / v
            d.xfrc_applied[self.body, 1] = -f * vg[1] / v

        for _ in range(self.substeps):
            mujoco.mj_step(self.m, d)
        self.t += dt
        self._read()
