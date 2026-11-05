# Open questions (GPS/map stack)

Questions to ask before the kart runs. Each one changes something in the sim
(`gps.yaml`) or in the race setup. Tick them off with the answer and the date.

## For the AKS / Purdue organizers

The organizers' RCS system, as they described it:
- **RCS Server:** runs the track state machine and has an RTK GPS base station.
- **RCS Transponder:** a black box on each kart, with its own battery and an onboard RTK rover. It talks to the server over LoRa, possibly as a mesh, using MQTT.
- **RCS HW Connector:** a software package on the kart computer that exchanges UDP packets with the transponder over Ethernet. There is also a web dashboard for flags.

So there is an RTK base on site.

- [ ] **Can karts drive on the transponder's RTK position,** or is it only for timing and flags? If we can use it, we might not need our own GNSS.
- [ ] **Transponder position output:**
  - rate in Hz;
  - delay;
  - what is reported (fixed/float status, accuracy, velocity, heading);
  - message format over UDP.
  - The sim assumes 20 Hz with 50 ms delay; change `sensors.gnss` if this is different.
- [ ] **Can teams get the base station's corrections (RTCM)** to feed our own ZED-F9P? Over what link? LoRa is probably too slow for full RTCM; it might go over the track Wi-Fi instead.
- [ ] **The base station's coordinates and frame,** and whether they are surveyed accurately or set by survey-in. Our map comes from the 2021 RTK survey, which used a base about 2.7 km away (-86.92619664, 40.45790787). Both must agree, or the map is shifted. The sim currently assumes they agree (map offset 0).
- [ ] **Do the organizers have their own surveyed track map** (edges, start line, pit lane)? It would be in the same frame as their base.
- [ ] **Practice time to drive or walk the track** with RTK, to check the map against their base.
- [ ] **Off-track rule:** what counts as off track (all four wheels out, or any wheel past the edge line), and what is the penalty? The sim ends the lap if any part of any tyre crosses the edge, which is stricter than most rules.
- [ ] **Pit-lane openings:** must karts stay inside the surveyed edge across the two pit openings? The sim treats the edge as closed there.
- [ ] **Race start:** a standing start from the grid or a rolling start? Where on the grid, and which way does the kart face? The sim uses a standing start with the rear axle on the checkered line.
- [ ] **Race direction** is assumed counter-clockwise.
- [ ] **Speed limits** in practice or the race, if any. The sim uses the kart's own limit of 12 m/s.
- [ ] **Track Wi-Fi** (Ubiquiti Rocket + Bullets, set up by Jack S. / UCSD): what can karts send over it, and with what bandwidth and delay? Could it carry RTK corrections?
- [ ] **What data are we allowed to bring on the kart?** The whole GPS approach depends on the answer.
  - A pre-built track map, for example our 2021 RTK survey of the edges.
  - A pre-computed racing line and speed plan.
  - Maps or lines recorded during practice laps.
  - Trained models (the ML driver) and logged data from practice.
  - Is anything required to be built live on the kart during the race? Does anything have to be shown or submitted beforehand?
  - Can we receive anything from off the kart during a race (corrections, remote commands, telemetry), or must the kart be self-contained apart from the RCS flags?

## For the team

- [ ] **GNSS:** the BOM's ArduSimple Skylark Starter Kit (ZED-F9P, about $553) isn't bought yet. Decide after the organizers answer whether the transponder can be used.
  - **The kit** is the hardware on the kart:
    - the receiver board (u-blox ZED-F9P);
    - the antenna;
    - the 4G modem that fetches corrections over the internet.
    - On its own, without corrections, it is only accurate to about 1–2 m.
  - **The Skylark subscription** is a paid internet service from Swift Navigation that sends the corrections. They come from Swift's own network of base stations, so we don't need a base. It gets the receiver to centimetre level.
    - About $69/month for cm-level and $29/month for decimetre-level (2024 prices). The kit includes a 6-month free trial.
    - Needs mobile signal at the track.
  - **We need both,** or the kit plus another correction source: the organizers' on-site base, or a free state NTRIP network (Indiana has one, INCORS).
  - The kit is a one-off purchase; the subscription is monthly and only needed while we use Skylark.
  - The free trial covers 6 months. Check that it overlaps the practice days and the race.
  - **The map frame must match.** Skylark positions are in a global frame. The 2021 survey's base frame is unknown (possibly NAD83, or a base set by survey-in). These can differ by about 1–1.5 m in Indiana.
    - Before the race, measure a few known points at the track (start-line corners, edge points) with the kit, and shift the map to match. That gets it to about ±5 cm.
    - The sim models this as a map offset of ±0.05 m.
- [ ] **GNSS antenna:** a mast above the rear-axle centre, about 0.75 m off the ground, is assumed. Is that OK with the bodywork and the motor/VESC noise?
- [ ] **Kart mass** is assumed to be 125 kg (the rule minimum). The BOM adds up to about 115 kg, so about 10 kg of ballast is needed. Weigh the kart when built, including the front/rear split (45/55 assumed) and the height of the centre of mass (0.2 m assumed).
- [ ] **Steering motor:** maximum rate (80°/s assumed), delay (40 ms assumed) and the lock angle at the wheels (±25° assumed). Is there a steering angle sensor?
- [ ] **Brake actuator:**
  - stroke speed (10 mm/s from the BOM);
  - travel from the bite point to full pressure (0.6 s assumed);
  - can it be held at the bite point while driving (assumed yes)?
  - is there a pressure sensor?
- [ ] **Drive and regen:**
  - VESC current limits;
  - gearing;
  - motor top speed (12 m/s assumed);
  - drive force (500 N assumed);
  - battery charge-current limit for regen (about 60 A / 3.5 kW assumed).
- [ ] **Wheel encoder:** where it sits (rear axle assumed) and its resolution.
- [ ] **IMU (BNO055):** mounting spot (near the centre of mass assumed). Will the magnetometer be used? It is probably unreliable near the motor, so the sim uses only the gyro and accelerometer.
- [ ] **Tyres:** any grip data, or a skidpad / coast-down test once the kart runs, to fit:
  - grip (`mu`);
  - rear slip stiffness (`slip_ratio_peak`);
  - steering jacking (`steer_jacking`);
  - rolling and drag resistance.

### kart-sw contracts (raised by the local mock, 2026-11-03)

The local mock (`C:\AKS RL\kart-sw-mock`) had to invent these; kart-sw has not decided them.
- [ ] **Command boundary:** what the Jetson sends the MCU/VESCs: motor current, torque, throttle/brake or speed? The mock sends motor force (N), brake (0–1) and steering angle (rad).
- [ ] **Wheel encoder topic:** kart-sw lists no encoder topic. The mock uses `sensors/wheel/speed` (m/s).
- [ ] **GNSS quality:** how RTK fixed vs float reaches ROS. The mock uses `NavSatFix.status` (GBAS_FIX = fixed) plus a separate `sensors/gnss/vel` velocity topic.
- [ ] **Estimator output:** `nav_msgs/Odometry` has no acceleration or GNSS quality. The mock adds one `EstimatorState` message for the controllers.
- [ ] **Trajectory message:** the mock sends the closed racing line (x, y, speed) plus the track centreline with edge room (`TrackBounds`).
- [ ] **Scaffold bug to report:** the ament_python package manifests list `ament_python` as a `buildtool_depend`. It isn't a rosdep key, so `rosdep install` fails and none of the scaffold images can build. The mock skips it with `--skip-keys ament_python`.

## Answered

- 2026-10-29: map offset. The base station is on site and calibrated, so the sim's map offset is 0 and only GNSS error remains. A stress test is kept: `scripts/gps/benchmark.py --map-shift 0.3`. Still to confirm: the organizers' base frame matches the 2021 survey (see above).
