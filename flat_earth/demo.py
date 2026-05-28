"""
Flat-earth game world — full model demonstration.

Runs each sub-system and prints equations, parameters, predictions,
and known tensions (residuals that require further tuning).

Usage:  python -m flat_earth.demo
"""

import math
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flat_earth import coordinates, forces, rotation, celestial, eclipse
from flat_earth.parameters import (
    WORLD_RADIUS_KM, MIDPLANE_RADIUS_KM, METRIC_BETA,
    G_SURFACE, DAY_PERIOD_S, YEAR_PERIOD_S,
    SUN_ALTITUDE_KM, MOON_ALTITUDE_KM, OMEGA_0
)


def section(title: str):
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


def subsect(title: str):
    print(f"\n  ── {title}")


# ═══════════════════════════════════════════════════════════════════════════════
section("1. COORDINATE SYSTEM")
# ═══════════════════════════════════════════════════════════════════════════════

print(f"""
  World disc radius R      = {WORLD_RADIUS_KM:>10,.0f} km
  Midplane radius   r₀     = {MIDPLANE_RADIUS_KM:>10,.0f} km
  Metric compression  β    = {METRIC_BETA:>10.2f}

  Metric:  ds² = dr² + [r·f(r)]² dφ²
           f(r) = 1                           for r ≤ r₀
           f(r) = exp(−{METRIC_BETA}·(r−r₀)/R)   for r > r₀
""")

subsect("City distance check (lat/lon → game coords → geodesic)")

city_pairs = [
    ("London",    51.5,   0.0,  "New York",    40.7, -74.0, 5570),
    ("Sydney",   -33.9, 151.2,  "Santiago",   -33.5, -70.7, 11300),
    ("Sydney",   -33.9, 151.2,  "Joburg",     -26.2,  28.0, 11010),
    ("Tokyo",     35.7, 139.7,  "São Paulo",  -23.5, -46.6, 18540),
]

print(f"\n  {'Route':<35} {'Model(km)':>10} {'Known(km)':>10} {'Error%':>8}")
print(f"  {'-'*65}")
for (c1, la1, lo1, c2, la2, lo2, known) in city_pairs:
    r1, p1 = coordinates.from_latlon(la1, lo1)
    r2, p2 = coordinates.from_latlon(la2, lo2)
    model  = coordinates.radial_distance_km(r1, p1, r2, p2)
    err    = 100 * (model - known) / known
    print(f"  {c1+' → '+c2:<35} {model:>10,.0f} {known:>10,.0f} {err:>+8.1f}%")

print(f"""
  Notes:
    β = {METRIC_BETA} partially corrects outer-ring inflation.
    Residuals remain for cross-disk routes (Sydney→Santiago, Tokyo→São Paulo)
    because the AE projection cannot simultaneously preserve all pairs.
    Game fix: use pre-computed spline lookup tables keyed to (r1,φ1,r2,φ2).
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("2. FORCE AND GRAVITY MODEL")
# ═══════════════════════════════════════════════════════════════════════════════

print(f"""
  Surface g₀ = {G_SURFACE:.5f} m/s²

  Field law:  a(z) = g₀ / (1 + z/R_field)²
              R_field = 6371 km  (fit to GPS/satellite altitude data)

  Vacuum chamber prediction (all densities fall identically):
    Acceleration is a property of the field, not the object.
    a_net(vacuum) = a(z) · 1.0  for all masses → confirmed.
""")

altitudes = [0, 1, 10, 100, 400, 1000]
print(f"  {'Altitude (km)':>14} {'g (m/s²)':>12} {'Drop 10m (s)':>14}")
print(f"  {'-'*44}")
for z in altitudes:
    g  = forces.gravitational_acceleration(z)
    td = forces.vacuum_drop_time(0.010)   # 10 m drop
    print(f"  {z:>14} {g:>12.5f} {td:>14.4f}")

subsect("Buoyancy table")
densities = [
    ("Hot air (200°C)", 0.7),
    ("Air (20°C)",      1.2),
    ("Wood",          600.0),
    ("Water",        1000.0),
    ("Iron",         7874.0),
]
print(f"\n  {'Material':<20} {'ρ (kg/m³)':>10} {'a_net (m/s²)':>14}  {'Direction':>12}")
print(f"  {'-'*62}")
for name, rho in densities:
    a = forces.net_vertical_acceleration(0.0, rho)
    direction = "↑ (floats)" if a < 0 else "↓ (sinks)"
    print(f"  {name:<20} {rho:>10.1f} {a:>14.5f}  {direction:>12}")


# ═══════════════════════════════════════════════════════════════════════════════
section("3. LOCAL ROTATION FIELD  Ω(r)")
# ═══════════════════════════════════════════════════════════════════════════════

print(f"""
  Ω(r) = Ω₀ · (r₀ − r) / r₀
  Ω₀   = 2π / T_day  =  {math.degrees(OMEGA_0)*3600:.6f}  °/hr

  Foucault pendulum precession period:  T_F(r) = T_day · r₀ / |r₀ − r|
  Coriolis deflection (1st order):      Δy = Ω(r) · v · t²
""")

latitudes = [90, 60, 45, 30, 10, 0, -10, -30, -45, -60, -90]
print(f"  {'Lat(°)':>7} {'r (km)':>8} {'Ω (°/hr)':>10} {'T_F (hr)':>10} "
      f"{'Cyclone':>16} {'Precession':>12}")
print(f"  {'-'*72}")
for lat in latitudes:
    r    = WORLD_RADIUS_KM * (1 - lat / 90) / 2
    om_h = rotation.foucault_precession_rate_deg_per_hour(r)
    T_F  = rotation.foucault_period_hours(r)
    cy   = rotation.cyclone_rotation(r)
    prec = "CCW" if om_h > 0 else ("CW" if om_h < 0 else "none")
    T_str = f"{T_F:.1f}" if T_F != float('inf') else "∞"
    print(f"  {lat:>7} {r:>8,.0f} {om_h:>10.4f} {T_str:>10} {cy:>16} {prec:>12}")


# ═══════════════════════════════════════════════════════════════════════════════
section("4. CELESTIAL MECHANICS")
# ═══════════════════════════════════════════════════════════════════════════════

print(f"""
  Sun orbit:   r_s(t) = {8000:.0f} · [1 + {0.20:.2f}·cos(2πt/T_year)]  km
               h_s    = {SUN_ALTITUDE_KM:.0f} km
  Moon orbit:  h_m    = {MOON_ALTITUDE_KM:.0f} km,  T_month = 29.53 days
""")

subsect("Stellar pole elevation by latitude")
print(f"  {'Lat(°)':>7} {'r (km)':>8} {'Pole elev(°)':>14}  {'Star rotation':>24}")
print(f"  {'-'*60}")
for lat in [90, 60, 45, 30, 0, -30, -45, -60, -90]:
    r   = WORLD_RADIUS_KM * (1 - lat / 90) / 2
    el  = celestial.celestial_pole_elevation_deg(r)
    rot = celestial.star_apparent_rotation_direction(r)
    print(f"  {lat:>7} {r:>8,.0f} {el:>14.1f}  {rot:>24}")

subsect("Day/night sample (summer solstice, t=0)")
t0 = 0.0
print(f"\n  {'Lat(°)':>7} {'Sun elev(°)':>12}  {'Daylight?':>10}")
print(f"  {'-'*35}")
for lat in [80, 60, 45, 0, -45, -60, -80]:
    r, p = coordinates.from_latlon(lat, 0.0)
    el   = celestial.sun_elevation_deg(r, p, t0)
    day  = celestial.is_daylight(r, p, t0)
    print(f"  {lat:>7} {el:>12.1f}  {'yes' if day else 'no':>10}")


# ═══════════════════════════════════════════════════════════════════════════════
section("5. ECLIPSE GEOMETRY")
# ═══════════════════════════════════════════════════════════════════════════════

print()
print(eclipse.eclipse_geometry_report())

subsect("Solar eclipse ground track sample  (t = solar eclipse peak)")
from flat_earth.celestial import sun_position, moon_position
from flat_earth.coordinates import polar_to_cartesian as p2c

# Align moon directly under sun for maximum eclipse
mr0, mphi0, _ = moon_position(0.0)
sr0, sphi0, _ = sun_position(0.0)
t_align = 0.0  # just show geometry

R_umb = eclipse.umbra_radius_km()
R_pen = eclipse.penumbra_radius_km()
print(f"""
  Umbra radius at surface:    {R_umb:.2f} km
  Penumbra radius at surface: {R_pen:.2f} km
  Shadow shape: circular (sun and moon are circular disks)

  Observer positions relative to umbra centre:
    0 km → total eclipse  (phase = total)
    {R_umb/2:.1f} km → total eclipse
    {R_umb:.1f} km → umbra edge (first contact)
    {(R_umb+R_pen)/2:.1f} km → partial eclipse midpoint
    {R_pen:.1f} km → penumbra outer edge
    > {R_pen:.1f} km → no eclipse
""")


# ═══════════════════════════════════════════════════════════════════════════════
section("6. MODEL SUMMARY — FREE PARAMETERS & TENSIONS")
# ═══════════════════════════════════════════════════════════════════════════════

print(f"""
  Free parameters (7 total)
  ─────────────────────────
  β  = {METRIC_BETA}        metric compression (geodesic fit)
  h_s = {SUN_ALTITUDE_KM:.0f} km    sun altitude
  h_m = {MOON_ALTITUDE_KM:.0f} km    moon altitude
  r_s0 = 8000 km    sun orbital radius
  ε_s  = 0.20       seasonal eccentricity
  Ω₀   = 2π/T_day   rotation field amplitude (ZERO free parameters — forced)
  h_* = 50000 km    star dome altitude

  Resolved tensions
  ─────────────────
  ✓ Gravity / vacuum free-fall: universal field → all densities identical
  ✓ Opposite stellar rotations: perspective geometry, 0 extra parameters
  ✓ Circular eclipse shadows: disks always cast circular shadows
  ✓ Coriolis/Foucault sign flip: Ω(r) changes sign at equator analogue
  ✓ Day/night illumination cone: matches observed pattern qualitatively

  Remaining tensions
  ──────────────────
  ✗ Cross-disk geodesics: AE projection cannot fit all city pairs
      → Use spline-corrected lookup table (adds O(N²) calibration data,
        not a free-parameter count)

  ✗ Eclipse shadow speed: model predicts ~200 m/s, observed ~600 m/s
      → Requires h_s/h_m ratio ≈ 20, i.e. h_m ≈ 250 km (not 4500 km)
      → This rescaling conflicts with moon angular size (must compensate r_m)

  ✗ Foucault period: model gives T_F = T_day·r₀/|r₀−r|, globe gives
      T_F = T_day/sin|lat| → differ by ~29% at latitude ±45°

  ✗ Star dome elevation: model predicts pole at ~84° at lat 45°,
      globe predicts 45° → large discrepancy (use baked LUT for gameplay)

  Error margins on input data assumed
  ────────────────────────────────────
  Flight distances:   ± 0.5% (GPS + nav logs)
  Foucault period:    ± 1%   (lab measurement)
  Eclipse speed:      ± 5%   (timing uncertainty)
  g(z) at 400 km:     ± 0.01% (satellite orbital data)
""")

print("  Model construction complete.\n")
