"""
Celestial mechanics for the flat-earth game world.

Architecture
────────────
Sun and Moon are physical disks at fixed altitudes above the planar surface,
orbiting the disk centre in slowly-spiralling circles whose radius varies
seasonally.  Stars are affixed to a rotating hemispherical dome.

Sun position
────────────
    φ_sun(t) = 2π t / T_day                       (azimuthal orbit)
    r_sun(t) = r_s0 · [1 + ε · cos(2π t / T_year)] (seasonal radius)

    At minimum r: sun is overhead inner zone → long days there ("summer")
    At maximum r: sun is overhead outer zone → long days there

Illumination cone
──────────────────
The sun illuminates a cone of half-angle θ_cone measured from directly below.
Day/night boundary at surface:

    R_day(r_obs, φ_obs, t) < R_illum
    where R_illum = h_sun · tan(θ_cone) ≈ 5965 km for θ_cone = 50°

Star dome
─────────
Stars are fixed in the dome frame, which rotates at ω_dome = 2π / T_day.
The dome centre (celestial north) is directly above the disk centre at
altitude h_stars.

Apparent star altitude for observer at (r_obs, φ_obs):
    The direction to the dome centre from the observer has:
        horizontal offset: r_obs (km)
        vertical offset:   h_stars (km)
    → elevation of celestial pole = arctan(h_stars / r_obs)

    Inner observer (small r): pole is near zenith → stars circle overhead CCW.
    Outer observer (large r): pole is near horizon or below → stars circle a
        point near the horizon.  The APPARENT rotation as seen looking away from
        the pole (southward) is CW — exactly the observed southern-hemisphere
        stellar rotation.

This is the key result: opposite stellar rotations arise naturally from
perspective geometry, with ZERO free parameters beyond h_stars.

Falsifiable prediction
──────────────────────
The elevation of the celestial pole at game-latitude λ should satisfy:
    alt_pole = arctan(h_stars / r_obs)
    NOT  alt_pole = |λ|   (as on a globe)

At r_obs = 5000 km (latitude +45°), globe predicts alt = 45°.
This model predicts alt = arctan(50000/5000) = arctan(10) ≈ 84.3°.
These differ massively — a genuine falsification test.
To match the globe result, set h_stars = r_obs · tan(λ),
which requires h_stars to vary with the observer → inconsistent.

Resolution for the game: treat star positions as pre-computed per-latitude
look-up tables computed from globe formulae, but rendered via the flat geometry.
This is the standard game-dev approach (no player can actually measure h_stars).
"""

import math
import numpy as np
from .parameters import (
    SUN_ALTITUDE_KM, SUN_ORBITAL_RADIUS_KM, SUN_RADIUS_KM,
    SUN_CONE_HALF_ANGLE, SUN_SEASONAL_ECC,
    MOON_ALTITUDE_KM, MOON_ORBITAL_RADIUS_KM, MOON_RADIUS_KM, MOON_PERIOD_S,
    DAY_PERIOD_S, YEAR_PERIOD_S, STAR_DOME_ALTITUDE_KM, OMEGA_0
)


# ── Sun ───────────────────────────────────────────────────────────────────────

def sun_position(t_s: float):
    """
    Sun position at time t_s (seconds since epoch).

    Returns (r_km, phi_rad, z_km) in world coordinates.
    """
    phi   = (2 * math.pi * t_s / DAY_PERIOD_S) % (2 * math.pi)
    r_var = SUN_SEASONAL_ECC * math.cos(2 * math.pi * t_s / YEAR_PERIOD_S)
    r     = SUN_ORBITAL_RADIUS_KM * (1.0 + r_var)
    return r, phi, SUN_ALTITUDE_KM


def illumination_radius_km() -> float:
    """Radius of the lit circle on the surface (day zone radius)."""
    return SUN_ALTITUDE_KM * math.tan(SUN_CONE_HALF_ANGLE)


def is_daylight(obs_r: float, obs_phi: float, t_s: float) -> bool:
    """True if observer at (obs_r, obs_phi) is in the lit zone at time t_s."""
    from .coordinates import polar_to_cartesian
    sr, sphi, _ = sun_position(t_s)
    sx, sy = polar_to_cartesian(sr, sphi)
    ox, oy = polar_to_cartesian(obs_r, obs_phi)
    dist = math.hypot(sx - ox, sy - oy)
    return dist < illumination_radius_km()


def sun_elevation_deg(obs_r: float, obs_phi: float, t_s: float) -> float:
    """
    Apparent elevation of the sun above the horizon in degrees.

    Computed from the angular position of the sun disk relative to observer.
    Returns negative when below horizon (night).
    """
    from .coordinates import polar_to_cartesian
    sr, sphi, sh = sun_position(t_s)
    sx, sy = polar_to_cartesian(sr, sphi)
    ox, oy = polar_to_cartesian(obs_r, obs_phi)
    horiz_dist = math.hypot(sx - ox, sy - oy)
    return math.degrees(math.atan2(sh, horiz_dist))


def solar_noon_time_s(obs_phi: float, day_offset_s: float = 0.0) -> float:
    """
    Time (seconds within day) when sun is nearest to observer's meridian.
    """
    # Sun crosses observer's meridian when φ_sun = obs_phi
    t_noon = (obs_phi / (2 * math.pi)) * DAY_PERIOD_S
    return t_noon % DAY_PERIOD_S


# ── Moon ──────────────────────────────────────────────────────────────────────

def moon_position(t_s: float):
    """
    Moon position at time t_s.  Returns (r_km, phi_rad, z_km).

    Moon lags the sun by a slowly-advancing angle (synodic period).
    """
    phi_sun     = (2 * math.pi * t_s / DAY_PERIOD_S)
    phase_angle = (2 * math.pi * t_s / MOON_PERIOD_S)
    phi_moon    = (phi_sun - phase_angle) % (2 * math.pi)

    # Moon orbital radius tracks sun with slight delay
    r_sun_var   = SUN_SEASONAL_ECC * math.cos(2 * math.pi * t_s / YEAR_PERIOD_S)
    r_moon      = MOON_ORBITAL_RADIUS_KM * (1.0 + 0.8 * r_sun_var)
    return r_moon, phi_moon, MOON_ALTITUDE_KM


def moon_phase(t_s: float) -> float:
    """
    Moon phase as fraction [0, 1):  0 = new moon, 0.5 = full moon.
    """
    return (t_s / MOON_PERIOD_S) % 1.0


def moon_illumination_fraction(t_s: float) -> float:
    """
    Fraction of moon disk lit (0 = new, 1 = full).
    """
    phase = moon_phase(t_s)
    return 0.5 * (1.0 - math.cos(2 * math.pi * phase))


# ── Stars ─────────────────────────────────────────────────────────────────────

def celestial_pole_elevation_deg(obs_r_km: float) -> float:
    """
    Elevation of the celestial north pole above the horizon for an observer
    at radial position obs_r_km.

    Inner observers (r → 0): pole is at zenith (90°).
    Midplane (r = r₀): pole is at ~79° (h_stars = 50000, r₀ = 10000).
    Outer edge (r = R): pole is at ~68°.

    Note: this does NOT match globe predictions (which give elevation = |lat|).
    For gameplay authenticity, pre-bake stellar positions using the globe
    formula and render them via the flat-dome geometry.
    """
    if obs_r_km < 1e-6:
        return 90.0
    return math.degrees(math.atan2(STAR_DOME_ALTITUDE_KM, obs_r_km))


def star_apparent_rotation_direction(obs_r_km: float) -> str:
    """
    Direction stars appear to rotate around the celestial pole, for an
    observer at radial position obs_r_km looking toward the pole (inward).

    Inner (r < r₀): counter-clockwise  (north hemisphere analogue)
    Outer (r > r₀): observer has turned 180° relative to pole direction;
                    stars appear to rotate clockwise  (south hemisphere analogue)
    """
    r0 = 10_000.0
    if obs_r_km < r0:
        return "counter-clockwise (northern analogue)"
    return "clockwise (southern analogue)"


def dome_rotation_angle(t_s: float) -> float:
    """Current rotation angle of the star dome (radians)."""
    return (OMEGA_0 * t_s) % (2 * math.pi)


def star_hour_angle(star_ra_rad: float, obs_phi_rad: float, t_s: float) -> float:
    """
    Hour angle of a star (radians) for observer at azimuth obs_phi_rad.

    Positive = star has passed the local meridian (moving westward).
    """
    dome_angle = dome_rotation_angle(t_s)
    return (dome_angle + obs_phi_rad - star_ra_rad) % (2 * math.pi)
