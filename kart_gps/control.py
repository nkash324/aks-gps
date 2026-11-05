"""Low-level control: turns a driver command (target speed m/s, curvature 1/m) into
motor force, brake and steering angle. Runs at control.lowlevel_hz (100 Hz).

It only uses what the real kart can measure (estimated speed, rear-axle encoder
speed) and the NOMINAL kart numbers, never the randomised ones, so the same class runs
on the Jetson. Braking uses regen first (free energy, no actuator lag) and adds the
hydraulic brake for whatever regen can't do, with a simple anti-lock check.
"""
import math

G = 9.81
AIR = 1.2
BRAKE_DEADBAND = 40.0     # N: don't touch the mechanical brake for less than this


class LowLevel:
    def __init__(self, kart, ctrl):
        """kart: nominal gps.yaml 'kart' dict. ctrl: gps.yaml 'control' dict."""
        self.k = kart
        self.c = ctrl
        L = kart["wheelbase"]
        t_in = math.tan(math.radians(kart["max_steer_deg"]))
        self.max_delta = math.atan(t_in * L / (L + kart["kingpin_half"] * t_in))
        self.reset()

    def reset(self):
        self.v_target = 0.0
        self.curvature = 0.0
        self.integ = 0.0
        self.brake_scale = 1.0
        self.drive_scale = 1.0

    def set_target(self, speed, curvature):
        self.v_target = max(0.0, float(speed))
        self.curvature = float(curvature)

    def steer_angle(self, curvature):
        """Bicycle steering angle for a path curvature at the rear axle."""
        return max(-self.max_delta, min(self.max_delta, math.atan(self.k["wheelbase"] * curvature)))

    def update(self, v, axle_speed, dt, lat_accel=0.0):
        """v: estimated kart speed, axle_speed: rear wheel speed from the encoder (m/s),
        lat_accel: IMU sideways acceleration (m/s^2).
        -> (motor force N, brake 0..1, steering angle rad)."""
        k, c = self.k, self.c
        m = k["mass"]
        err = self.v_target - v
        a = c["speed_kp"] * err + self.integ
        a = max(-c["max_brake_decel"], min(4.0, a))
        # integrate only when not saturated (anti-windup)
        if -c["max_brake_decel"] < a < 4.0:
            self.integ = max(-1.0, min(1.0, self.integ + c["speed_ki"] * err * dt))
        if self.v_target < 0.05 and v < 0.3:
            self.integ = 0.0
        resist = k["rolling_resistance"] * m * G + 0.5 * AIR * k["drag_area"] * v * v
        force = m * a + resist

        slip = (axle_speed - v) / max(v, 1.0)
        if slip < -c["abs_slip"]:
            self.brake_scale = max(0.2, self.brake_scale * 0.6)
        else:
            self.brake_scale = min(1.0, self.brake_scale + 0.1)
        if slip > c["abs_slip"]:
            self.drive_scale = max(0.2, self.drive_scale * 0.6)
        else:
            self.drive_scale = min(1.0, self.drive_scale + 0.1)

        if force >= 0.0:
            # friction circle: while cornering hard, the rears have little grip left for
            # driving; full throttle there spins the kart
            a_grip = c["grip_accel"]
            room = math.sqrt(max(0.0, a_grip ** 2 - lat_accel ** 2)) * m
            motor, brake = min(force, resist + room) * self.drive_scale, 0.0
        else:
            regen_max = min(k["regen_force"], k["regen_power"] / max(axle_speed, 0.1)) * min(1.0, axle_speed)
            motor = -min(-force, regen_max) * self.brake_scale
            rest = -force - regen_max
            brake = 0.0
            if rest > BRAKE_DEADBAND or (self.v_target < 0.05 and v < 1.0):
                brake = min(1.0, max(rest, 0.0) / k["brake_force"]) * self.brake_scale
                if self.v_target < 0.05 and v < 1.0:
                    brake = 1.0              # hold the kart at a stop
        return motor, brake, self.steer_angle(self.curvature)
