from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np

from .trajectory import AISTrajectory

if TYPE_CHECKING:
    from camar.maps.enc_map import ENCProjection


class AISReplayPolicy:
    """Policy that replays pre-recorded AIS trajectories as agent actions.

    Each trajectory is mapped to one agent. At every ``env.step()`` call the
    policy returns the pre-computed displacement action for the current
    timestep. Shorter trajectories are **padded with zero actions** so agents
    remain stationary after their track ends.

    This policy is designed for use with :class:`~camar.dynamics.DeltaPosDynamic`.

    Parameters
    ----------
    trajectories:
        One :class:`~camar.ais.AISTrajectory` per agent. Trajectories should
        already be interpolated to a uniform timestep matching the environment's
        ``step_dt``.
    projection:
        :class:`~camar.maps.enc_map.ENCProjection` instance used to convert
        lon/lat positions to km simulator coordinates.
    max_speed_km_per_step:
        Must match ``DeltaPosDynamic.max_speed``. Used to normalise displacements
        to the ``[-1, 1]`` action range.
    warn_clipping:
        If ``True``, emit a warning when any displacement exceeds
        ``max_speed_km_per_step`` (i.e. action is clipped).

    Attributes
    ----------
    n_agents : int
    n_steps : int
        Length of the longest trajectory (total policy horizon).
    t_grid : np.ndarray or None
        Unix-second timestamp for every step, set when trajectories share a
        common time grid (e.g. produced by ``align_trajectories_to_window``).
    """

    def __init__(
        self,
        trajectories: list[AISTrajectory],
        projection: "ENCProjection",
        max_speed_km_per_step: float,
        max_angle_delta_rad: float = 0.5,
        warn_clipping: bool = True,
    ):
        if not trajectories:
            raise ValueError("trajectories must be non-empty")

        self.n_agents = len(trajectories)
        self.max_speed = max_speed_km_per_step
        self.max_angle_delta = max_angle_delta_rad

        # Store the global time grid if all trajectories share the same timestamps
        first_ts = trajectories[0].timestamps
        if all(len(t.timestamps) == len(first_ts) and
               np.allclose(t.timestamps, first_ts, atol=1.0) for t in trajectories):
            self.t_grid: Optional[np.ndarray] = first_ts.copy()
        else:
            self.t_grid = None

        # Project all trajectories to km simulator coordinates
        xy_seqs: list[np.ndarray] = []
        for traj in trajectories:
            xs, ys = projection.forward(traj.lons, traj.lats)
            xy_seqs.append(np.stack([xs, ys], axis=1).astype(np.float32))  # (T_i, 2)

        # Pad to the same length (longest trajectory) with the last position
        lengths = [s.shape[0] for s in xy_seqs]
        self.n_steps = max(lengths)

        padded: list[np.ndarray] = []
        for seq in xy_seqs:
            if seq.shape[0] < self.n_steps:
                pad = np.tile(seq[-1:], (self.n_steps - seq.shape[0], 1))
                seq = np.concatenate([seq, pad], axis=0)
            padded.append(seq)

        # positions shape: (n_steps, n_agents, 2)
        positions = np.stack(padded, axis=1)

        # Pre-compute displacement actions: delta[t] moves agent from t → t+1
        # Shape: (n_steps, n_agents, 2)  — last step is zero (already at final pos)
        deltas = np.zeros_like(positions)
        deltas[:-1] = positions[1:] - positions[:-1]

        # Normalise position deltas to [-1, 1] action space
        pos_actions = deltas / max_speed_km_per_step  # (n_steps, n_agents, 2)

        # Warn if any position step exceeds the max speed
        if warn_clipping:
            max_action = np.abs(pos_actions).max()
            if max_action > 1.0:
                n_clipped = int((np.abs(pos_actions) > 1.0).any(axis=-1).sum())
                warnings.warn(
                    f"{n_clipped} steps exceed max_speed_km_per_step={max_speed_km_per_step} "
                    f"(max displacement ratio = {max_action:.2f}). "
                    "Actions will be clipped to [-1, 1]. "
                    "Consider increasing max_speed or reducing interpolation interval.",
                    UserWarning,
                    stacklevel=2,
                )

        # ------------------------------------------------------------------
        # Build heading (delta) actions — 3rd action dimension
        # ------------------------------------------------------------------
        heading_seqs: list[np.ndarray] = []
        for traj in trajectories:
            angles = np.radians(traj.headings_deg).astype(np.float64)  # (T_i,)
            if len(angles) < self.n_steps:
                pad = np.full(self.n_steps - len(angles), angles[-1])
                angles = np.concatenate([angles, pad])
            heading_seqs.append(angles)

        headings = np.stack(heading_seqs, axis=1)  # (n_steps, n_agents)

        dh = np.zeros_like(headings)
        dh[:-1] = headings[1:] - headings[:-1]
        dh = np.arctan2(np.sin(dh), np.cos(dh))   # wrap to [-π, π]
        heading_actions = dh / max_angle_delta_rad  # normalise to [-1, 1]

        # Combine: (n_steps, n_agents, 3)
        actions = np.concatenate([pos_actions, heading_actions[:, :, np.newaxis]], axis=-1)
        self._actions = np.clip(actions, -1.0, 1.0).astype(np.float32)
        # (n_steps, n_agents, 3)

        # Initial headings (radians) for env initialisation
        self._initial_angles = np.array(
            [np.radians(traj.headings_deg[0]) for traj in trajectories], dtype=np.float32
        )  # (n_agents,)

        # Store start and end km positions for env initialisation
        self._initial_positions = positions[0].copy()   # (n_agents, 2)
        # Goal = last real (non-padded) position of each trajectory
        self._goal_positions = np.array(
            [xy_seqs[i][-1] for i in range(self.n_agents)], dtype=np.float32
        )  # (n_agents, 2)

    def __call__(self, obs, state) -> np.ndarray:
        """Return actions for the current timestep.

        Parameters
        ----------
        obs:
            Current observation (unused — trajectory replay ignores obs).
        state:
            Current simulator state. ``state.step`` determines which
            pre-computed action to return.

        Returns
        -------
        np.ndarray
            Shape ``(n_agents, 2)``, values in ``[-1, 1]``.
        """
        step = int(state.step)
        if step >= self.n_steps:
            # Beyond trajectory horizon — all agents stay still
            return np.zeros((self.n_agents, 2), dtype=np.float32)
        return self._actions[step]

    def initial_positions(self) -> np.ndarray:
        """Starting km coordinates for all agents (first point of each trajectory).

        Returns
        -------
        np.ndarray
            Shape ``(n_agents, 2)`` in the simulator's km coordinate system.
        """
        return self._initial_positions.copy()

    def goal_positions(self) -> np.ndarray:
        """Goal km coordinates for all agents (last point of each trajectory).

        Returns
        -------
        np.ndarray
            Shape ``(n_agents, 2)`` in the simulator's km coordinate system.
        """
        return self._goal_positions.copy()

    def initial_angles(self) -> np.ndarray:
        """Initial heading (radians) for all agents (first point of each trajectory).

        Returns
        -------
        np.ndarray
            Shape ``(n_agents,)`` in radians, 0 = north, clockwise positive.
        """
        return self._initial_angles.copy()

    def __repr__(self) -> str:
        return (
            f"AISReplayPolicy(n_agents={self.n_agents}, n_steps={self.n_steps}, "
            f"max_speed={self.max_speed} km/step, "
            f"max_angle_delta={self.max_angle_delta:.3f} rad/step)"
        )
