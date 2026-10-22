"""V2X-Radar-C loader for REAL 4-D radar detections (both agents).

Frames (verified on the data, see PROJECT_STATUS):
  * labels of both agents are already in the common (RSU lidar) frame;
  * radar points are in the radar frame; x_to_world(yaml['radar']['cords']) maps
    them to the common frame (identical to lidar_pose @ radar.extrinsic);
  * the ego radar's boresight is its own +y axis;
  * the 'dopple' field is the radial velocity RELATIVE to the radar (positive =
    receding), NOT ego-motion compensated; yaml 'ego_speed' is unreliable (0 while
    moving), so the radar velocity is taken from finite differences of 'cords'.

Association: the radar points of a vehicle are those inside its annotated box
(+ margin). This is ORACLE association -- the labels pick the points, the radar
supplies the measured values. Data association is outside the paper's scope and
the paper must say so.
"""
from __future__ import annotations
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data/v2x_radar/V2X-Radar-C"
AGENTS = ("-1", "142")            # stationary RSU, ego vehicle
DT = 0.1
XY_MARGIN = 0.5                   # m, added to the box half-extents
Z_MARGIN = 1.0


def read_pcd(path):
    b = open(path, "rb").read()
    i = b.index(b"DATA binary\n") + len(b"DATA binary\n")
    hdr = {l.split()[0]: l.split()[1:] for l in b[:i].decode().splitlines() if l and not l.startswith("#")}
    tmap = {("F", "4"): "<f4", ("F", "8"): "<f8", ("U", "4"): "<u4", ("I", "4"): "<i4"}
    dt = np.dtype([(f, tmap[(t, s)]) for f, t, s in zip(hdr["FIELDS"], hdr["TYPE"], hdr["SIZE"])])
    n = int(hdr["POINTS"][0])
    return np.frombuffer(b[i:i + n * dt.itemsize], dt)


def x_to_world(p):
    """OpenCOOD convention: p = [x, y, z, roll, yaw, pitch] (degrees)."""
    x, y, z, r, yw, pi = p
    r, yw, pi = np.radians([r, yw, pi])
    cy, sy, cr, sr, cp, sp = np.cos(yw), np.sin(yw), np.cos(r), np.sin(r), np.cos(pi), np.sin(pi)
    m = np.eye(4)
    m[:3, 3] = [x, y, z]
    m[0] = [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x]
    m[1] = [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y]
    m[2] = [sp, -cp * sr, cp * cr, z]
    return m


def scenes(splits=("train", "validate", "test")):
    return sorted(p for s in splits for p in (ROOT / s).iterdir()
                  if p.is_dir() and (p / "-1").is_dir() and (p / "142").is_dir())


def load_scene(sdir):
    """Per frame: labels (pooled, common frame), radar pose/velocity, radar points."""
    frames = sorted(f.stem for f in (sdir / "-1").glob("*.yaml"))
    Y = {a: [yaml.safe_load(open(sdir / a / f"{f}.yaml")) for f in frames] for a in AGENTS}
    T = len(frames)
    pose = {a: np.array([Y[a][t]["radar"]["cords"][:2] for t in range(T)], float) for a in AGENTS}
    vel = {a: np.gradient(pose[a], DT, axis=0) for a in AGENTS}
    return dict(sdir=sdir, frames=frames, Y=Y, T=T, radar_pos=pose, radar_vel=vel)


def labels_at(sc, t):
    """{vid: (agent-visibility tuple, mean location xyz, half extents, yaw)} at frame t."""
    out = {}
    for j, a in enumerate(AGENTS):
        for k, v in (sc["Y"][a][t].get("vehicles") or {}).items():
            loc = np.array(v["location"], float)
            if not np.all(np.isfinite(loc)):
                continue
            if k in out:
                vis, locs, ext, yw = out[k]
                vis = tuple(True if i == j else vis[i] for i in range(2))
                out[k] = (vis, locs + [loc], ext, yw)
            else:
                vis = tuple(i == j for i in range(2))
                out[k] = (vis, [loc], np.array(v["extent"], float), np.radians(v["angle"][1]))
    return {k: (vis, np.mean(locs, 0), ext, yw) for k, (vis, locs, ext, yw) in out.items()}


def radar_points(sc, a, t):
    """Radar points of agent a at frame t in the common frame, with ego-compensated Doppler."""
    d = sc["Y"][a][t]
    r = read_pcd(sc["sdir"] / a / f"{sc['frames'][t]}_radar.pcd")
    M = x_to_world(d["radar"]["cords"])
    P = (M @ np.c_[r["x"], r["y"], r["z"], np.ones(len(r))].astype(float).T).T[:, :3]
    dk = [k for k in r.dtype.names if k.startswith("dop")][0]
    rel = P[:, :2] - sc["radar_pos"][a][t]
    o = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-9)
    dop = r[dk].astype(float) + o @ sc["radar_vel"][a][t]     # radial velocity w.r.t. the ground
    return P, dop


def measure(P, dop, sensor_xy, loc, ext, yaw):
    """Oracle-associated measurement [azimuth, range, range-rate] from the points in the box."""
    d = P[:, :2] - loc[:2]
    u = d @ [np.cos(yaw), np.sin(yaw)]
    w = d @ [-np.sin(yaw), np.cos(yaw)]
    h = (np.abs(u) < ext[0] + XY_MARGIN) & (np.abs(w) < ext[1] + XY_MARGIN) & \
        (np.abs(P[:, 2] - loc[2]) < ext[2] + Z_MARGIN)
    n = int(h.sum())
    if n == 0:
        return None, 0
    rel = P[h, :2] - sensor_xy
    rng = np.linalg.norm(rel, axis=1)
    az = np.angle(np.mean(np.exp(1j * np.arctan2(rel[:, 1], rel[:, 0]))))
    return np.array([az, np.median(rng), np.median(dop[h])]), n
