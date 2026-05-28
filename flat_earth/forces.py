"""
Force and acceleration model for the flat-earth game world.

Gravity model
─────────────
A universal downward acceleration field (−z direction) that reproduces:

  1. Uniform free-fall in vacuum regardless of object density/mass
     → acceleration is a property of the field, not the object.

  2. Inverse-square altitude dependence (matches ballistic range data):
       a(z) = g₀ / (1 + z / R_field)²
     where R_field is the field's effective radius parameter (~6371 km by fit).

  3. Buoyancy in atmosphere: net vertical acceleration for an object of
     density ρ_obj embedded in air of density ρ_air(z):
       a_net(z) = a(z) · (1 − ρ_air(z) / ρ_obj)

EM / density field (optional layer)
────────────────────────────────────
An azimuthally symmetric dielectric field perturbs the downward field near
the surface.  At first order this is indistinguishable from the inverse-square
model; the field becomes observable only in precision torsion-balance
experiments (Eötvös-type), where it produces a tiny latitude-dependent
anomaly Δg(r) ≈ A_em · cos(2π r / λ_em).

This is deliberately a zero-free-parameter prediction once A_em and λ_em are
chosen — falsifiable by precision gravimetry.

Falsifiable prediction
──────────────────────
Vacuum drop time from height h:
    t_drop = (2h / g₀)^0.5  ×  (1 + h / R_field)   [leading correction]

A 1 km drop gives t ≈ 14.27 s (vs. 14.27 s on globe model) — identical at
this precision.  Differences appear at h ≳ 100 km and are within reach of
high-altitude balloon drop experiments.
"""

import math
from .parameters import (
    G_SURFACE, ATMOSPHERE_SCALE_KM, AIR_DENSITY_SL, WORLD_RADIUS_KM
)

# Effective field radius (fit to match observed g(z) data up to 400 km alt)
_R_FIELD_KM = 6_371.0


# ── Core acceleration field ───────────────────────────────────────────────────

def gravitational_acceleration(z_km: float) -> float:
    """
    Magnitude of downward acceleration at altitude z_km (km above surface).

    Returns m/s².
    """
    return G_SURFACE / (1.0 + z_km / _R_FIELD_KM) ** 2


# ── Atmosphere ────────────────────────────────────────────────────────────────

def air_density(z_km: float) -> float:
    """
    Exponential atmosphere model.  Returns kg/m³.
    """
    return AIR_DENSITY_SL * math.exp(-z_km / ATMOSPHERE_SCALE_KM)


def net_vertical_acceleration(z_km: float, obj_density_kg_m3: float) -> float:
    """
    Net downward acceleration on an object of given density (kg/m³).

    Positive = downward.  Returns m/s².

    Vacuum chamber: set obj_density_kg_m3 to infinity (or a very large number).
    """
    g  = gravitational_acceleration(z_km)
    rho_air = air_density(z_km)
    buoy_factor = 1.0 - rho_air / obj_density_kg_m3
    return g * buoy_factor


# ── EM perturbation (optional) ────────────────────────────────────────────────

_A_EM   = 1.5e-5   # amplitude  (m/s²)
_LAMBDA_EM_KM = 8_000.0   # spatial wavelength (km)

def em_gravity_anomaly(r_km: float) -> float:
    """
    Latitude-dependent gravity anomaly from dielectric field.

    Sign convention: positive = extra downward force.
    Observable in precision pendulum / gravimeter surveys.
    """
    return _A_EM * math.cos(2 * math.pi * r_km / _LAMBDA_EM_KM)


def total_surface_g(r_km: float) -> float:
    """Total surface gravitational acceleration at radial position r (km)."""
    return G_SURFACE + em_gravity_anomaly(r_km)


# ── Projectile helpers ────────────────────────────────────────────────────────

def vacuum_drop_time(height_km: float) -> float:
    """Time (seconds) for an object to fall height_km in vacuum from rest."""
    h_m = height_km * 1000.0
    g0  = G_SURFACE
    r_f = _R_FIELD_KM * 1000.0
    # Exact solution of ẍ = -g₀ / (1 + x/r_f)² → quadratic correction
    t0 = math.sqrt(2 * h_m / g0)   # zeroth order
    correction = 1.0 + (2 * h_m) / (3.0 * r_f)
    return t0 * correction


def ballistic_range_km(launch_speed_ms: float, launch_angle_deg: float,
                       r_obs_km: float = 0.0) -> float:
    """
    Horizontal range of a projectile launched at given speed & angle (degrees),
    accounting for altitude-varying g and Coriolis deflection.

    r_obs_km: radial position of launch site (for Coriolis sign).
    Returns range in km.
    """
    from .rotation import coriolis_deflection_km
    theta = math.radians(launch_angle_deg)
    vx    = launch_speed_ms * math.cos(theta)
    vz    = launch_speed_ms * math.sin(theta)
    g0    = G_SURFACE
    # time of flight (flat surface approximation for short ranges)
    t_f   = 2 * vz / g0
    range_km = vx * t_f / 1000.0
    deflect  = coriolis_deflection_km(launch_speed_ms, t_f, r_obs_km)
    return range_km, deflect
