from __future__ import annotations

from typing import Optional

import numpy as np

from .trajectory import AISTrajectory

_KM_PER_KNOT_PER_S = 1.852 / 3600.0  # knots → km/s


def interpolate_trajectory(
    traj: AISTrajectory,
    interval_s: float = 10.0,
    method: str = "linear",
) -> AISTrajectory:
    """Resample a trajectory to a uniform time grid.

    Parameters
    ----------
    traj:
        Source trajectory (may have irregular timestamps).
    interval_s:
        Target time step in seconds.
    method:
        ``"linear"`` — linear interpolation on each coordinate independently.
        ``"cubic_hermite"`` — cubic Hermite spline fitted to positions using
        speed + course as tangent vectors at each knot.

    Returns
    -------
    AISTrajectory
        New trajectory with uniform ``interval_s`` timestep.
        ``speeds_kn`` and ``courses_deg`` are always linearly interpolated.
    """
    if method == "linear":
        return _interpolate_linear(traj, interval_s)
    elif method == "cubic_hermite":
        return _interpolate_cubic_hermite(traj, interval_s)
    else:
        raise ValueError(f"Unknown interpolation method '{method}'. Use 'linear' or 'cubic_hermite'.")


def _uniform_timestamps(traj: AISTrajectory, interval_s: float) -> np.ndarray:
    t0, t1 = traj.timestamps[0], traj.timestamps[-1]
    n = max(2, int(np.ceil((t1 - t0) / interval_s)) + 1)
    return np.linspace(t0, t1, n)


def _interpolate_linear(traj: AISTrajectory, interval_s: float) -> AISTrajectory:
    t_new = _uniform_timestamps(traj, interval_s)
    t_old = traj.timestamps

    lons_new = np.interp(t_new, t_old, traj.lons)
    lats_new = np.interp(t_new, t_old, traj.lats)
    speeds_new = np.interp(t_new, t_old, traj.speeds_kn)
    courses_new = _interp_angle(t_new, t_old, traj.courses_deg)
    headings_new = _interp_angle(t_new, t_old, traj.headings_deg)

    return AISTrajectory(
        mmsi=traj.mmsi,
        timestamps=t_new,
        lons=lons_new,
        lats=lats_new,
        speeds_kn=speeds_new,
        courses_deg=courses_new,
        headings_deg=headings_new,
        ship_type=traj.ship_type,
    )


def _interpolate_cubic_hermite(traj: AISTrajectory, interval_s: float) -> AISTrajectory:
    try:
        from scipy.interpolate import CubicHermiteSpline
    except ImportError as e:
        raise ImportError(
            "cubic_hermite interpolation requires scipy. "
            "Install with: mamba install -c conda-forge scipy"
        ) from e

    t_old = traj.timestamps
    t_new = _uniform_timestamps(traj, interval_s)

    # Tangent vectors at each knot derived from speed + COG
    # COG is degrees clockwise from north: x = sin(cog), y = cos(cog)
    course_rad = np.radians(traj.courses_deg)
    speed_km_s = traj.speeds_kn * _KM_PER_KNOT_PER_S

    # Convert km/s velocity to approximate degrees/s for lon/lat derivatives
    # d_lon/dt ≈ (vx_km_s) / (111.32 * cos(lat_rad))
    # d_lat/dt ≈ (vy_km_s) / 111.32
    cos_lat = np.cos(np.radians(traj.lats))
    vx_km_s = speed_km_s * np.sin(course_rad)   # eastward km/s
    vy_km_s = speed_km_s * np.cos(course_rad)   # northward km/s

    d_lon_dt = vx_km_s / (111.32 * np.clip(cos_lat, 1e-6, None))
    d_lat_dt = vy_km_s / 111.32

    lon_spline = CubicHermiteSpline(t_old, traj.lons, d_lon_dt)
    lat_spline = CubicHermiteSpline(t_old, traj.lats, d_lat_dt)

    lons_new = lon_spline(t_new)
    lats_new = lat_spline(t_new)
    speeds_new = np.interp(t_new, t_old, traj.speeds_kn)
    courses_new = _interp_angle(t_new, t_old, traj.courses_deg)
    headings_new = _interp_angle(t_new, t_old, traj.headings_deg)

    return AISTrajectory(
        mmsi=traj.mmsi,
        timestamps=t_new,
        lons=lons_new,
        lats=lats_new,
        speeds_kn=speeds_new,
        courses_deg=courses_new,
        headings_deg=headings_new,
        ship_type=traj.ship_type,
    )


def align_trajectories_to_window(
    trajectories: list[AISTrajectory],
    interval_s: float = 10.0,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    window_s: float = 7200.0,
    min_active_fraction: float = 0.05,
    method: str = "linear",
) -> tuple[list[AISTrajectory], float, float]:
    """Resample all trajectories onto a single shared global time grid.

    Every output trajectory has the same ``timestamps`` array (the global grid),
    so step *i* corresponds to the same real-world instant for every vessel.

    Ships that are not yet active (or already gone) at a given grid step are
    held at their trajectory's nearest endpoint and their ``speeds_kn`` is set
    to zero for that step.

    Parameters
    ----------
    trajectories:
        Raw extracted trajectories (irregular timestamps, any length).
    interval_s:
        Grid step in seconds.
    t_start, t_end:
        Unix-second boundaries of the shared window.  When *None* the function
        slides a ``window_s``-wide window across the data and picks the position
        that maximises the number of active vessels.
    window_s:
        Width (seconds) of the auto-detected window. Default 2 hours.
    min_active_fraction:
        Trajectories with fewer than this fraction of steps inside their real
        time range are excluded from the returned list.  Default 0.05 (5 %).

    Returns
    -------
    aligned_trajectories:
        One trajectory per kept input, all sharing the same ``timestamps``.
    t_start:
        Start of the chosen window (Unix seconds).
    t_end:
        End of the chosen window (Unix seconds).
    """
    if not trajectories:
        raise ValueError("trajectories list is empty")

    starts = np.array([t.timestamps[0] for t in trajectories])
    ends   = np.array([t.timestamps[-1] for t in trajectories])

    # ------------------------------------------------------------------
    # Auto-detect the window_s-wide interval with the most active ships
    # ------------------------------------------------------------------
    if t_start is None or t_end is None:
        day_start = float(starts.min())
        day_end   = float(ends.max())
        search_step = max(interval_s, 60.0)  # coarse scan, ≥1 minute
        best_ws, best_count = day_start, 0
        ws = day_start
        while ws + window_s <= day_end:
            we = ws + window_s
            active = int(((starts <= we) & (ends >= ws)).sum())
            if active > best_count:
                best_count = active
                best_ws = ws
            ws += search_step
        t_start = best_ws
        t_end   = best_ws + window_s
        print(f"align_trajectories_to_window: best {window_s/3600:.1f}h window "
              f"[{_unix_to_str(t_start)} – {_unix_to_str(t_end)}] "
              f"with {best_count} active vessels")

    # ------------------------------------------------------------------
    # Build the global time grid
    # ------------------------------------------------------------------
    n_steps = max(2, int(round((t_end - t_start) / interval_s)) + 1)
    t_grid  = np.linspace(t_start, t_end, n_steps)

    # ------------------------------------------------------------------
    # Resample each trajectory onto t_grid
    # ------------------------------------------------------------------
    aligned: list[AISTrajectory] = []
    for traj in trajectories:
        t_lo, t_hi = traj.timestamps[0], traj.timestamps[-1]
        active_mask = (t_grid >= t_lo) & (t_grid <= t_hi)

        if active_mask.mean() < min_active_fraction:
            continue  # too little real data in this window

        # Clamp query times to traj's own range so np.interp extrapolates
        # to the nearest endpoint (ship stays still outside its range).
        t_q = np.clip(t_grid, t_lo, t_hi)

        lons  = np.interp(t_q, traj.timestamps, traj.lons)
        lats  = np.interp(t_q, traj.timestamps, traj.lats)
        speeds = np.interp(t_q, traj.timestamps, traj.speeds_kn)
        speeds = np.where(active_mask, speeds, 0.0)  # zero speed outside active range
        courses  = _interp_angle(t_q, traj.timestamps, traj.courses_deg)
        headings = _interp_angle(t_q, traj.timestamps, traj.headings_deg)

        aligned.append(AISTrajectory(
            mmsi=traj.mmsi,
            timestamps=t_grid.copy(),
            lons=lons,
            lats=lats,
            speeds_kn=speeds,
            courses_deg=courses,
            headings_deg=headings,
            ship_type=traj.ship_type,
        ))

    return aligned, float(t_start), float(t_end)


def _unix_to_str(t: float) -> str:
    """Format a Unix timestamp as 'YYYY-MM-DD HH:MM:SS'."""
    import datetime
    return datetime.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _interp_angle(t_new: np.ndarray, t_old: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Interpolate angles (degrees) via unit-vector decomposition to avoid wrap artefacts."""
    rad = np.radians(angles_deg)
    sin_interp = np.interp(t_new, t_old, np.sin(rad))
    cos_interp = np.interp(t_new, t_old, np.cos(rad))
    return np.degrees(np.arctan2(sin_interp, cos_interp)) % 360.0
