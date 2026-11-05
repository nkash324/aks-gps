"""GPS/map driving stack: 2D kart physics, sensors + EKF, planner, drivers.

The track comes from tracks/purdue_rtk/ (built from the 2021 RTK edge survey).
Nothing here needs MuJoCo, so the estimator, map and drivers run on the Jetson as-is.
"""
