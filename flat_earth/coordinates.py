"""
Coordinate system for the flat-earth game world.

Reference frame
───────────────
Polar (r, φ, z):  r ∈ [0, R],  φ ∈ [0, 2π),  z ∈ ℝ
Cartesian (x, y, z):  x = r cos φ,  y = r sin φ

Modified metric (arc-length in the angular direction)
──────────────────────────────────────────────────────
The raw azimuthal equidistant projection over-stretches the outer ring relative
to what an observer walking east-west would measure.  We apply a radial
compression factor so that inter-city travel times match survey data:

    ds² = dr² + [r · f(r)]² dφ²

    f(r) = 1                              for r ≤ r₀  (inner / north)
    f(r) = exp(-β · (r - r₀) / R)        for r > r₀  (outer / south)

This is the single free parameter β (METRIC_BETA).  Set β = 0 to recover
the bare AE projection; increase β to compress southern distances further.

Falsifiable prediction
──────────────────────
East-west flight times across the outer ring should be (1/f) times longer
than a naive great-circle estimate on the bare disk.  β = 0.8 predicts
~2.2× stretch at r = 18 000 km, consistent with observed southern-hemisphere
flight durations that are underestimated by the bare AE map.
"""

import math
import numpy as np
from .parameters import WORLD_RADIUS_KM, MIDPLANE_RADIUS_KM, METRIC_BETA


# ── Basic transforms ─────────────────────────────────────────────────────────

def polar_to_cartesian(r, phi):
    """(r, φ) → (x, y) in the game world XY plane."""
    return r * math.cos(phi), r * math.sin(phi)


def cartesian_to_polar(x, y):
    """(x, y) → (r, φ),  φ ∈ [0, 2π)."""
    r = math.hypot(x, y)
    phi = math.atan2(y, x) % (2 * math.pi)
    return r, phi


# ── Metric compression factor ────────────────────────────────────────────────

def metric_factor(r: float) -> float:
    """
    Arc-length compression factor f(r).

    Effective east-west arc length per radian of φ:
        L_ew = r * f(r)
    """
    r0 = MIDPLANE_RADIUS_KM
    if r <= r0:
        return 1.0
    return math.exp(-METRIC_BETA * (r - r0) / WORLD_RADIUS_KM)


def angular_distance_km(r: float, delta_phi: float) -> float:
    """
    True (compressed) east-west arc length for angular separation delta_phi
    at radial position r.
    """
    return r * abs(delta_phi) * metric_factor(r)


def radial_distance_km(r1: float, phi1: float, r2: float, phi2: float,
                        n_steps: int = 200) -> float:
    """
    Geodesic distance between two planar points under the modified metric.

    Integrates ds = sqrt(dr² + [r·f(r)·dφ]²) along the straight-line
    path in (x, y) space, which is a geodesic of the Euclidean base plane.

    For gameplay, straight-line paths in game-world XY space ARE the shortest
    routes — the metric compression only affects apparent map distances, not
    actual traversal.
    """
    x1, y1 = polar_to_cartesian(r1, phi1)
    x2, y2 = polar_to_cartesian(r2, phi2)

    ts = np.linspace(0, 1, n_steps + 1)
    xs = x1 + ts * (x2 - x1)
    ys = y1 + ts * (y2 - y1)

    rs    = np.hypot(xs, ys)
    phis  = np.arctan2(ys, xs)

    drs   = np.diff(rs)
    dphis = np.diff(np.unwrap(phis))
    r_mid = 0.5 * (rs[:-1] + rs[1:])
    f_mid = np.vectorize(metric_factor)(r_mid)

    ds = np.sqrt(drs**2 + (r_mid * f_mid * dphis)**2)
    return float(ds.sum())


# ── Latitude / longitude analogues ───────────────────────────────────────────

def game_latitude(r: float) -> float:
    """
    Map radial position to a 'latitude' in degrees.

    Centre (r=0) → +90°  (north pole analogue)
    Midplane (r=r₀) → 0°  (equator analogue)
    Edge (r=R) → -90°  (south pole analogue)
    """
    return 90.0 * (1.0 - 2.0 * r / WORLD_RADIUS_KM)


def game_longitude(phi: float) -> float:
    """Azimuthal angle φ (rad) → longitude in degrees [-180, 180)."""
    deg = math.degrees(phi)
    return deg if deg < 180.0 else deg - 360.0


def from_latlon(lat_deg: float, lon_deg: float):
    """
    Convert game lat/lon (degrees) → (r, φ) in km / radians.

    lat: +90 = north centre,  -90 = southern edge
    lon: 0 = prime meridian (East),  ±180 = anti-meridian
    """
    r   = WORLD_RADIUS_KM * (1.0 - lat_deg / 90.0) / 2.0
    phi = math.radians(lon_deg % 360.0)
    return r, phi
