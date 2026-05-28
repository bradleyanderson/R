"""
World constants for the flat-earth game model.

The world is a disk of radius WORLD_RADIUS_KM, oriented in the XY plane.
The center (0, 0) corresponds to the "north pole" analogue.
The outer edge at r = WORLD_RADIUS_KM is the "southern boundary" / ice wall.
Elevation z increases upward.

All distances in kilometres unless noted. Time in seconds.
"""

import math

# ── World geometry ──────────────────────────────────────────────────────────
WORLD_RADIUS_KM       = 20_000.0   # disk radius
MIDPLANE_RADIUS_KM    = 10_000.0   # "equator" analogue (r₀)

# Metric compression: outer region feels smaller than it looks on the map.
# Angular arc-length correction factor for r > MIDPLANE_RADIUS_KM:
#   effective_arc = r * dφ * f(r),  f(r) = exp(-METRIC_BETA * (r - r₀) / R)
METRIC_BETA           = 0.8        # compression strength (dimensionless)

# ── Gravity ─────────────────────────────────────────────────────────────────
G_SURFACE             = 9.80665    # m/s²  (surface standard)
ATMOSPHERE_SCALE_KM   = 8.5        # density scale height for drag/buoyancy
AIR_DENSITY_SL        = 1.225      # kg/m³ at surface (sea level analogue)

# ── Celestial bodies ────────────────────────────────────────────────────────
SUN_ALTITUDE_KM       = 5_000.0    # sun height above surface
SUN_ORBITAL_RADIUS_KM = 8_000.0    # mean orbital radius from disk centre
SUN_RADIUS_KM         = 700.0      # physical radius (for eclipse geometry)
SUN_CONE_HALF_ANGLE   = math.radians(50.0)  # half-angle of lit cone
SUN_SEASONAL_ECC      = 0.20       # seasonal orbital eccentricity ε

MOON_ALTITUDE_KM      = 4_000.0    # lower than sun → umbra reaches surface
MOON_ORBITAL_RADIUS_KM = 8_200.0   # tracks the sun roughly
MOON_RADIUS_KM        = 350.0      # smaller than sun → total eclipses possible
MOON_PERIOD_S         = 29.53 * 86_400.0   # synodic month

DAY_PERIOD_S          = 86_400.0
YEAR_PERIOD_S         = 365.25 * 86_400.0

# ── Rotation field (Coriolis / Foucault) ────────────────────────────────────
# Ω(r) = Ω₀ * (r₀ - r) / r₀
# Positive = CCW (inner/north), negative = CW (outer/south)
OMEGA_0               = 2 * math.pi / DAY_PERIOD_S   # rad/s  (one rot/day)

# ── Star dome ────────────────────────────────────────────────────────────────
STAR_DOME_ALTITUDE_KM = 50_000.0   # height of celestial sphere
POLARIS_DECLINATION   = math.radians(89.35)   # nearly overhead at disk centre
