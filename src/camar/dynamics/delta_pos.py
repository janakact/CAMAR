from typing import Type

import jax.numpy as jnp
from flax import struct
from jax.typing import ArrayLike

from .base import BaseDynamic, PhysicalState
from camar.registry import register_dynamic


@struct.dataclass
class DeltaPosState(PhysicalState):
    agent_pos: ArrayLike    # (num_agents, 2)
    agent_angle: ArrayLike  # (num_agents,)  heading in radians, 0 = north, CW

    @classmethod
    def create(
        cls,
        key: ArrayLike,
        landmark_pos: ArrayLike,
        agent_pos: ArrayLike,
        goal_pos: ArrayLike,
        agent_angle: ArrayLike,
        sizes: "Sizes",  # noqa: F821
    ) -> "DeltaPosState":
        return cls(agent_pos=agent_pos, agent_angle=agent_angle)


@register_dynamic("DeltaPosDynamic")
class DeltaPosDynamic(BaseDynamic):
    """
    Kinematic dynamic where the action is a normalised position displacement.

    Each call to ``integrate()`` moves agents by::

        new_pos = old_pos + actions * max_speed

    where ``actions`` is in ``[-1, 1]^2`` and ``max_speed`` is the maximum
    km displacement per integrate step.

    Collision forces are accepted but ignored — position is commanded directly.
    The environment still detects and reports collisions.

    Recommended configuration for 10-second AIS replay::

        DeltaPosDynamic(max_speed=0.2, dt=10.0)
        # with frameskip=0 → step_dt = 10 s, max displacement = 200 m/step

    Parameters
    ----------
    max_speed : float
        Maximum km displacement per ``integrate()`` call when ``action = ±1``.
    dt : float
        Seconds per ``integrate()`` call (used only for ``step_dt`` calculation).
    max_angle_delta : float
        Maximum heading change (radians) per ``integrate()`` call when
        ``action[2] = ±1``.  Default 0.5 rad ≈ 28.6°/step.
    """

    def __init__(self, max_speed: float = 0.1, dt: float = 1.0, max_angle_delta: float = 0.5):
        self.max_speed = max_speed
        self._dt = dt
        self.max_angle_delta = max_angle_delta

    @property
    def action_size(self) -> int:
        return 3

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def state_class(self) -> Type[DeltaPosState]:
        return DeltaPosState

    def integrate(
        self,
        key: ArrayLike,
        force: ArrayLike,
        physical_state: DeltaPosState,
        actions: ArrayLike,
    ) -> DeltaPosState:
        new_pos = physical_state.agent_pos + actions[:, :2] * self.max_speed
        raw_angle = physical_state.agent_angle + actions[:, 2] * self.max_angle_delta
        # Wrap to [-π, π]
        new_angle = jnp.arctan2(jnp.sin(raw_angle), jnp.cos(raw_angle))
        return physical_state.replace(agent_pos=new_pos, agent_angle=new_angle)
