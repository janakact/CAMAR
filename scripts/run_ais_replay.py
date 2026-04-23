"""
AIS Trajectory Replay — animated SVG output
============================================

Loads real AIS vessel tracks from parquet files, aligns them to a common
real-world time window, runs them through the CAMAR simulator using
DeltaPosDynamic, and saves the episode as an animated SVG.

All vessels share the same global time grid so that step N always represents
the same UTC instant for every ship.  The current time is shown as an
animated clock in the SVG.

Usage
-----
    python scripts/run_ais_replay.py [options]

Examples
--------
    # 300 vessels, auto-detect best 2-hour window, 10 s step, 30 fps
    python scripts/run_ais_replay.py --num-agents 300 --fps 30

    # Explicit time window
    python scripts/run_ais_replay.py --t-start "2018-12-29 08:15:00" \\
        --t-end "2018-12-29 10:15:00"

    # Custom output path
    python scripts/run_ais_replay.py --output out/malacca.svg
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# make sure the package is importable when running from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run AIS trajectory replay and save animated SVG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data",
        nargs="+",
        default=["data/combined_data_260219.parquet"],
        metavar="PATH",
        help="One or more AIS parquet files.",
    )
    p.add_argument(
        "--enc-dir",
        default="encdata/",
        metavar="DIR",
        help="Directory containing ENC shapefiles (landPolygons.shp, zones.shp, labels.json).",
    )
    p.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=[103.4, 0.9, 104.3, 1.6],
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="WGS84 bounding box for filtering AIS data and defining map extent.",
    )
    p.add_argument(
        "--num-agents",
        type=int,
        default=4,
        metavar="N",
        help="Number of vessels (agents) to replay simultaneously.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=10.0,
        metavar="SECONDS",
        help="Interpolation timestep in seconds (= simulator step_dt).",
    )
    p.add_argument(
        "--window",
        type=float,
        default=7200.0,
        metavar="SECONDS",
        help="Width (seconds) of the auto-detected time window. Default 2 hours.",
    )
    p.add_argument(
        "--t-start",
        default=None,
        metavar="DATETIME",
        help="Start of replay window as 'YYYY-MM-DD HH:MM:SS' UTC. "
             "Overrides auto-detection.",
    )
    p.add_argument(
        "--t-end",
        default=None,
        metavar="DATETIME",
        help="End of replay window as 'YYYY-MM-DD HH:MM:SS' UTC. "
             "Overrides auto-detection.",
    )
    p.add_argument(
        "--max-speed",
        type=float,
        default=0.5,
        metavar="KM",
        help="Max km displacement per step (DeltaPosDynamic.max_speed). "
             "0.5 km / 10 s ≈ 97 knots — safely covers all vessel speeds.",
    )
    p.add_argument(
        "--max-angle-delta",
        type=float,
        default=0.5,
        metavar="RAD",
        help="Max heading change per step in radians (DeltaPosDynamic.max_angle_delta). "
             "0.5 rad ≈ 28.6°/step.",
    )
    p.add_argument(
        "--agent-rad",
        type=float,
        default=0.8,
        metavar="KM",
        help="Agent arrow size in km.",
    )
    p.add_argument(
        "--min-speed",
        type=float,
        default=1.0,
        metavar="KNOTS",
        help="Minimum average speed (knots) over active steps to include a vessel. "
             "Filters out anchored ships.",
    )
    p.add_argument(
        "--min-active",
        type=float,
        default=0.05,
        metavar="FRACTION",
        help="Minimum fraction of window steps a vessel must be active (0–1).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of simulation steps. Defaults to the full window.",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Frames per second for the SVG animation.",
    )
    p.add_argument(
        "--min-points",
        type=int,
        default=30,
        metavar="N",
        help="Minimum raw AIS points for a trajectory to be considered.",
    )
    p.add_argument(
        "--max-gap",
        type=float,
        default=120.0,
        metavar="SECONDS",
        help="Max time gap (s) before splitting a vessel track into a new segment.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="JAX random seed.",
    )
    p.add_argument(
        "--output",
        default="ais_replay.svg",
        metavar="FILE",
        help="Output SVG path.",
    )
    p.add_argument(
        "--coastline-sampling",
        type=float,
        default=0.5,
        metavar="KM",
        help="Coastline sampling density for landmark circles (km).",
    )
    return p.parse_args()


def _parse_datetime(s: str) -> float:
    """Parse 'YYYY-MM-DD HH:MM:SS' UTC string → Unix seconds."""
    import datetime
    dt = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=datetime.timezone.utc).timestamp()


def main() -> None:
    args = parse_args()

    import jax
    import jax.numpy as jnp
    from camar import camar_v0
    from camar.ais import (
        AISReplayPolicy,
        align_trajectories_to_window,
        extract_trajectories,
        load_ais_parquet,
    )
    from camar.dynamics import DeltaPosDynamic
    from camar.maps import enc_map
    from camar.render import SVGVisualizer

    bbox = tuple(args.bbox)

    # ------------------------------------------------------------------
    # 1. Load AIS data
    # ------------------------------------------------------------------
    print(f"Loading AIS data from: {args.data}")
    df = load_ais_parquet(args.data, bbox=bbox)
    print(f"  {len(df):,} rows, {df['ais_static_mmsi'].nunique()} unique vessels in bbox")

    # ------------------------------------------------------------------
    # 2. Extract raw trajectories
    # ------------------------------------------------------------------
    print(f"\nExtracting trajectories (min_points={args.min_points}, max_gap={args.max_gap}s) ...")
    raw_trajs = extract_trajectories(df, min_points=args.min_points, max_gap_s=args.max_gap)
    print(f"  {len(raw_trajs)} segments found")

    # ------------------------------------------------------------------
    # 3. Align all trajectories to a shared time window
    # ------------------------------------------------------------------
    t_start_unix = _parse_datetime(args.t_start) if args.t_start else None
    t_end_unix   = _parse_datetime(args.t_end)   if args.t_end   else None

    print(f"\nAligning to common time window "
          f"(interval={args.interval}s, window={args.window/3600:.1f}h, "
          f"min_active={args.min_active:.0%}) ...")
    aligned, t_start_unix, t_end_unix = align_trajectories_to_window(
        raw_trajs,
        interval_s=args.interval,
        t_start=t_start_unix,
        t_end=t_end_unix,
        window_s=args.window,
        min_active_fraction=args.min_active,
    )
    print(f"  {len(aligned)} trajectories active in window")

    # ------------------------------------------------------------------
    # 4. Filter and select vessels
    # ------------------------------------------------------------------
    # Speed filter: compute mean speed only over active (non-clamped) steps
    # Active steps have non-zero speed in the aligned trajectory
    def active_mean_speed(traj):
        active = traj.speeds_kn > 0
        if active.sum() == 0:
            return 0.0
        return float(traj.speeds_kn[active].mean())

    # Filter by minimum active speed
    aligned = [t for t in aligned if active_mean_speed(t) >= args.min_speed]
    print(f"  {len(aligned)} after min_speed={args.min_speed} kn filter")

    # Filter out ships with no heading data (always heading == 0 degrees)
    import numpy as _np
    aligned = [t for t in aligned if not _np.all(t.headings_deg == 0)]
    print(f"  {len(aligned)} after removing always-zero-heading vessels")

    if len(aligned) < args.num_agents:
        print(
            f"  WARNING: only {len(aligned)} trajectories pass filters, "
            f"requested {args.num_agents}. Reducing num_agents."
        )
        args.num_agents = len(aligned)

    # Sort by active duration descending — prefer ships with more coverage
    aligned.sort(key=active_mean_speed, reverse=True)
    trajs = aligned[:args.num_agents]

    import datetime
    for i, traj in enumerate(trajs):
        t0_str = datetime.datetime.utcfromtimestamp(traj.timestamps[0]).strftime("%H:%M:%S")
        t1_str = datetime.datetime.utcfromtimestamp(traj.timestamps[-1]).strftime("%H:%M:%S")
        spd = active_mean_speed(traj)
        n_active = int((traj.speeds_kn > 0).sum())
        print(f"  [{i:3d}] MMSI {traj.mmsi:>12}  active {n_active:4d}/{traj.n_points} steps  "
              f"avg_speed={spd:.1f} kn  ship_type={traj.ship_type}")

    # ------------------------------------------------------------------
    # 5. Build ENC map + environment
    # ------------------------------------------------------------------
    args.num_agents = len(trajs)
    print(f"\nSelected {args.num_agents} moving vessels.")
    print(f"\nBuilding enc_map (bbox={bbox}, coastline_sampling={args.coastline_sampling} km) ...")
    m = enc_map(
        args.enc_dir,
        num_agents=args.num_agents,
        bbox=bbox,
        coastline_sampling_km=args.coastline_sampling,
        agent_rad_km=args.agent_rad,
    )

    dynamic = DeltaPosDynamic(
        max_speed=args.max_speed,
        dt=args.interval,
        max_angle_delta=args.max_angle_delta,
    )
    env = camar_v0(m, dynamic=dynamic, frameskip=0)
    print(f"  step_dt={env.step_dt}s  obs_size={env.observation_size}")

    # ------------------------------------------------------------------
    # 6. Build replay policy
    # ------------------------------------------------------------------
    policy = AISReplayPolicy(
        trajs,
        m.projection,
        max_speed_km_per_step=args.max_speed,
        max_angle_delta_rad=args.max_angle_delta,
        warn_clipping=True,
    )

    # Pin agent start positions and goals
    m.set_fixed_positions(policy.initial_positions(), policy.goal_positions())
    print("  Agent starts and goals set from trajectory endpoints.")
    print(f"\n{policy}")

    n_steps = args.steps if args.steps is not None else policy.n_steps
    n_steps = min(n_steps, policy.n_steps)
    print(f"Running {n_steps} steps ...")

    # ------------------------------------------------------------------
    # 7. Run simulation
    # ------------------------------------------------------------------
    key = jax.random.key(args.seed)
    obs, state = env.reset(key)

    # Apply initial headings from AIS data (env.reset initialises to 0)
    init_angles = jnp.array(policy.initial_angles(), dtype=jnp.float32)
    state = state.replace(physical_state=state.physical_state.replace(agent_angle=init_angles))

    state_seq = [state]
    for step_i in range(n_steps):
        actions = policy(obs, state)
        obs, state, _rew, _done, _ = env.step(key, state, actions)
        state_seq.append(state)
        if (step_i + 1) % 50 == 0:
            print(f"  step {step_i + 1}/{n_steps}")

    print(f"Simulation complete. Collected {len(state_seq)} frames.")

    # ------------------------------------------------------------------
    # 8. Save animated SVG
    # ------------------------------------------------------------------
    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    print(f"\nRendering SVG animation (fps={args.fps}) → {out_path}")

    # Pass global time grid so the SVG shows a real-time clock
    t_grid = policy.t_grid  # Unix-second timestamps, one per step
    SVGVisualizer(env, state_seq, fps=args.fps, t_grid=t_grid).save_svg(out_path)
    print("Done.")


if __name__ == "__main__":
    main()
