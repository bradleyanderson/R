"""
Eclipse mechanics for the flat-earth game world.

Solar eclipse
─────────────
The Moon passes between the Sun and the surface.  Both bodies are circular
disks (cross-section), so their cast shadows are always circular — this holds
regardless of planar or spherical geometry.

Umbra cone geometry:

    Sun at height h_s, radius r_s (physical radius in km)
    Moon at height h_m, radius r_m

    Apex of the umbra cone is at height h_apex above the surface:
        h_apex = h_m − r_m · (h_s − h_m) / (r_s − r_m)

    If h_apex > 0: cone intersects surface → total eclipse possible.
    If h_apex ≤ 0: only annular eclipse (ring of fire).

    Umbra radius at surface (h = 0):
        R_umbra = r_m · h_m / (h_s − h_m)   (for h_apex > 0)

    Penumbra radius at surface:
        R_penumbra = r_s · h_m / (h_s − h_m)

Both umbra and penumbra are perfect circles when sun & moon are concentric
(directly overhead).  At off-axis geometry (sun not directly above observer),
the shadow becomes an ellipse — identical behaviour to the spherical model.
Eclipse duration scales with umbra cross-speed at the surface.

Lunar eclipse (shadow object model)
─────────────────────────────────────
In this world model, the moon enters the "night cone" — the shadow region
where the sun's illumination cone does not reach.  The boundary of the night
cone at moon altitude h_m is a circle of radius:

    R_night_at_moon = R_illum · (1 − h_m / h_s)
                    = h_s · tan(θ_cone) · (1 − h_m / h_s)

The moon (radius r_m) begins to enter shadow when its centre is within:
    d < R_night_at_moon + r_m

Total darkness when:
    d < R_night_at_moon − r_m

The night-cone boundary at any given moment is a circle → the onset/exit
arc of the lunar eclipse shadow edge sweeps as a curved (circular) arc across
the moon disk, exactly as observed.

Falsifiable prediction
──────────────────────
Shadow speed at surface during solar eclipse:
    v_shadow = (moon orbital speed) × (h_s / (h_s − h_m))

With h_s = 5000, h_m = 4500, moon orbital circumference ≈ 2π × 8200 km
in T_month = 29.53 days:
    v_moon ≈ 2π × 8200 / (29.53 × 86400) ≈ 0.020 km/s = 20 m/s
    v_shadow ≈ 0.020 × (5000/500) = 0.20 km/s = 200 m/s

Observed eclipse shadow speed: ~0.5–1.0 km/s → off by ~4×.
To match: either increase h_s/h_m separation or increase moon orbital speed.
Parameter fit: h_s = 5000, h_m = 500 → v_shadow ≈ 0.020 × 10 = 0.20 km/s (still low).
Open tension: requires moon altitude ≪ sun altitude for realistic shadow speeds.
"""

import math
from .parameters import (
    SUN_ALTITUDE_KM, SUN_ORBITAL_RADIUS_KM, SUN_RADIUS_KM, SUN_CONE_HALF_ANGLE,
    MOON_ALTITUDE_KM, MOON_RADIUS_KM
)
from .celestial import sun_position, moon_position, illumination_radius_km
from .coordinates import polar_to_cartesian


# ── Solar eclipse ──────────────────────────────────────────────────────────────

def umbra_apex_height_km() -> float:
    """Height of the umbra cone apex above the surface (km)."""
    h_s, h_m = SUN_ALTITUDE_KM, MOON_ALTITUDE_KM
    r_s, r_m = SUN_RADIUS_KM, MOON_RADIUS_KM
    if r_s <= r_m:
        return float('inf')   # annular eclipse always
    return h_m - r_m * (h_s - h_m) / (r_s - r_m)


def umbra_radius_km() -> float:
    """
    Radius of the total solar eclipse umbra circle at the surface (km).

    Positive when total eclipse is possible (apex above surface).
    """
    h_apex = umbra_apex_height_km()
    if h_apex <= 0:
        return 0.0
    h_m, r_m, h_s = MOON_ALTITUDE_KM, MOON_RADIUS_KM, SUN_ALTITUDE_KM
    return r_m * h_m / (h_s - h_m)


def penumbra_radius_km() -> float:
    """Radius of the penumbra (partial shadow) at the surface (km)."""
    h_m, r_s, h_s = MOON_ALTITUDE_KM, SUN_RADIUS_KM, SUN_ALTITUDE_KM
    return r_s * h_m / (h_s - h_m)


def solar_eclipse_status(obs_r: float, obs_phi: float, t_s: float) -> dict:
    """
    Eclipse status for observer at (obs_r, obs_phi) at time t_s.

    Returns dict with keys: phase (str), obscuration (float 0-1),
    umbra_centre (x, y), umbra_radius_km, penumbra_radius_km.

    Phase: 'none', 'partial', 'total', 'annular'.
    """
    mr, mphi, _ = moon_position(t_s)
    sr, sphi, _ = sun_position(t_s)
    mx, my = polar_to_cartesian(mr, mphi)
    sx, sy = polar_to_cartesian(sr, sphi)
    ox, oy = polar_to_cartesian(obs_r, obs_phi)

    # Shadow centre projected to surface (moon directly below → shadow at mx,my)
    R_umb  = umbra_radius_km()
    R_pen  = penumbra_radius_km()

    dist_to_shadow = math.hypot(ox - mx, oy - my)

    if dist_to_shadow > R_pen:
        phase, obscuration = 'none', 0.0
    elif dist_to_shadow < R_umb:
        phase, obscuration = 'total', 1.0
    else:
        phase = 'partial'
        obscuration = (R_pen - dist_to_shadow) / (R_pen - R_umb + 1e-6)
        obscuration = max(0.0, min(1.0, obscuration))

    return {
        'phase': phase,
        'obscuration': obscuration,
        'umbra_centre': (mx, my),
        'umbra_radius_km': R_umb,
        'penumbra_radius_km': R_pen,
    }


# ── Lunar eclipse ─────────────────────────────────────────────────────────────

def night_cone_radius_at_moon_altitude_km() -> float:
    """
    Radius of the night cone (unlit zone) at the moon's altitude.
    """
    R_illum = illumination_radius_km()
    h_m = MOON_ALTITUDE_KM
    h_s = SUN_ALTITUDE_KM
    return R_illum * (1.0 - h_m / h_s)


def lunar_eclipse_status(t_s: float) -> dict:
    """
    Lunar eclipse status at time t_s.

    Returns dict with keys: phase (str), darkened_fraction (float 0-1).
    Phase: 'none', 'penumbral', 'partial', 'total'.
    """
    mr, mphi, _ = moon_position(t_s)
    sr, sphi, _ = sun_position(t_s)
    mx, my = polar_to_cartesian(mr, mphi)
    sx, sy = polar_to_cartesian(sr, sphi)

    # Distance from moon to sun (projected to moon altitude plane)
    dist = math.hypot(mx - sx, my - sy)

    R_nc  = night_cone_radius_at_moon_altitude_km()
    r_m   = MOON_RADIUS_KM

    if dist > R_nc + r_m:
        return {'phase': 'none', 'darkened_fraction': 0.0}
    elif dist > R_nc - r_m:
        fraction = (R_nc + r_m - dist) / (2 * r_m + 1e-6)
        return {'phase': 'partial', 'darkened_fraction': min(1.0, fraction)}
    else:
        return {'phase': 'total', 'darkened_fraction': 1.0}


# ── Geometry report ───────────────────────────────────────────────────────────

def eclipse_geometry_report() -> str:
    lines = [
        "Eclipse Geometry (flat-earth game model)",
        "=" * 42,
        f"Sun altitude:            {SUN_ALTITUDE_KM:>10.1f} km",
        f"Moon altitude:           {MOON_ALTITUDE_KM:>10.1f} km",
        f"Sun physical radius:     {SUN_RADIUS_KM:>10.1f} km",
        f"Moon physical radius:    {MOON_RADIUS_KM:>10.1f} km",
        f"Umbra apex height:       {umbra_apex_height_km():>10.1f} km",
        f"Umbra radius at surface: {umbra_radius_km():>10.2f} km",
        f"Penumbra radius:         {penumbra_radius_km():>10.2f} km",
        f"Night cone @ moon alt:   {night_cone_radius_at_moon_altitude_km():>10.1f} km",
        "",
        "Tension: shadow speed ≈ 200 m/s (observed: 500–1000 m/s).",
        "Fix: increase h_s/h_m altitude ratio, or rescale moon orbital speed.",
    ]
    return "\n".join(lines)
