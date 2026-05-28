"""
flat_earth — game world physics model
======================================
A self-consistent coordinate system and force model for a disk-world game.

Sub-modules
───────────
  parameters   — world constants (all distances in km, time in s)
  coordinates  — planar polar coordinate system with modified metric
  forces       — gravity field, atmosphere, buoyancy
  rotation     — Coriolis / Foucault rotation field Ω(r)
  celestial    — sun, moon, and star dome mechanics
  eclipse      — solar and lunar eclipse geometry

Quick-start
───────────
    from flat_earth import coordinates, celestial, eclipse, forces, rotation

    # Where is the sun right now?
    r, phi, z = celestial.sun_position(t_s=0)

    # Is the observer in daylight?
    daylight = celestial.is_daylight(obs_r=5000, obs_phi=0.3, t_s=3600*6)

    # Gravity at 10 km altitude
    g = forces.gravitational_acceleration(z_km=10)

    # Foucault pendulum at r=5000 km ("45° north")
    T = rotation.foucault_period_hours(r_km=5000)
"""

from . import parameters, coordinates, forces, rotation, celestial, eclipse

__all__ = ["parameters", "coordinates", "forces", "rotation", "celestial", "eclipse"]
