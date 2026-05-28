"""
Local rotation field — Coriolis and Foucault pendulum effects.

The world disk carries a radially-varying angular velocity field Ω(r).
This is the only mechanism that produces opposite circumpolar rotations
without a topological singularity.

Definition
──────────
    Ω(r) = Ω₀ · (r₀ − r) / r₀

    r₀ = MIDPLANE_RADIUS_KM  ("equator analogue")
    Ω₀ = 2π / T_day

Properties:
  • r = 0  (pole centre):  Ω = +Ω₀  →  CCW local rotation
  • r = r₀ (equator):      Ω = 0    →  no rotation
  • r = 2r₀ = R (edge):    Ω = −Ω₀  →  CW local rotation

Coriolis acceleration
─────────────────────
    a_C = −2 Ω(r) ẑ × v_horizontal

    In component form (vx, vy in the local east-north frame):
        a_Cx = +2 Ω(r) · vy
        a_Cy = −2 Ω(r) · vx

Foucault pendulum precession
────────────────────────────
    dφ/dt = −Ω(r)

    Precession period:  T_F(r) = 2π / |Ω(r)|  =  T_day · r₀ / |r₀ − r|

    At r = 0:  T_F = T_day  (identical to globe-model north pole)
    At r = r₀: T_F = ∞     (no precession, matches globe equator)
    At r = R:  T_F = T_day  (same magnitude, opposite sense)

Falsifiable prediction
──────────────────────
A Foucault pendulum at game-latitude −45° (r = 15 000 km) should precess
at dφ/dt = −Ω₀ · (10000 − 15000)/10000 = +Ω₀/2 → T_F = 2 days.
On a globe, sin(45°) = 0.707 → T_F ≈ 1.41 days.
The two models differ by ~29% at this latitude — a detectable discrepancy.
"""

import math
import numpy as np
from .parameters import OMEGA_0, MIDPLANE_RADIUS_KM, DAY_PERIOD_S


def local_omega(r_km: float) -> float:
    """
    Local angular velocity of the rotation field at radial position r (km).

    Units: rad/s.  Positive = CCW (inner/north), negative = CW (outer/south).
    """
    r0 = MIDPLANE_RADIUS_KM
    return OMEGA_0 * (r0 - r_km) / r0


def coriolis_acceleration(vx: float, vy: float, r_km: float):
    """
    Coriolis acceleration components (m/s²) for horizontal velocity (vx, vy)
    in the local East-North frame at radial position r_km.

    vx: eastward velocity (m/s)
    vy: northward velocity (toward disk centre) (m/s)

    Returns (ax, ay).
    """
    omega = local_omega(r_km)
    ax = +2.0 * omega * vy
    ay = -2.0 * omega * vx
    return ax, ay


def foucault_period_hours(r_km: float) -> float:
    """
    Foucault pendulum precession period (hours) at radial position r_km.

    Returns float('inf') at the equator analogue (r = r₀).
    """
    omega = local_omega(r_km)
    if abs(omega) < 1e-20:
        return float('inf')
    return abs(2 * math.pi / omega) / 3600.0


def foucault_precession_rate_deg_per_hour(r_km: float) -> float:
    """
    Foucault pendulum precession rate in degrees per hour.

    Positive = CCW, negative = CW.
    """
    omega = local_omega(r_km)
    return math.degrees(omega)


def coriolis_deflection_km(speed_ms: float, flight_time_s: float,
                            r_km: float) -> float:
    """
    Lateral deflection (km) of a projectile due to Coriolis, to first order.

    Positive = rightward deflection (inner/north hemisphere).
    """
    omega = local_omega(r_km)
    # Δy = Ω · v · t²  (standard first-order result)
    deflection_m = omega * speed_ms * flight_time_s ** 2
    return deflection_m / 1000.0


def wind_deflection_direction(r_km: float) -> str:
    """Human-readable deflection convention at radial position r_km."""
    omega = local_omega(r_km)
    if abs(omega) < 1e-12:
        return "none (equator analogue)"
    return "rightward (CCW cyclones)" if omega > 0 else "leftward (CW cyclones)"


def cyclone_rotation(r_km: float) -> str:
    """Low-pressure system rotation direction at r_km."""
    omega = local_omega(r_km)
    return "counter-clockwise" if omega > 0 else "clockwise"
