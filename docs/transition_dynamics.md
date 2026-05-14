# Transition Dynamics

This document describes the transition-dynamics layer of CAMAR: the abstraction that
maps `(state, action) → next_state` for every agent each simulation step. It covers
the existing dynamics, the proposed `inverse(state, next_state) → action` extension,
and the planned **sea-current** field that some dynamics will couple to.

---

## 1. Concept

A **dynamic** is a stateless object owned by the environment. It defines:

1. The shape of the per-agent **physical state** (position, velocity, heading, …).
2. The dimensionality of the per-agent **action**.
3. The integration step `f(state, action, force, key) → state`.
4. The simulator timestep `dt`.

The environment owns one `BaseDynamic` instance and one `PhysicalState` per episode.
On each `env.step`, the environment computes collision forces, calls
`dynamic.integrate(...)`, and stores the new state.

```
                    ┌──────────────────────┐
        actions ──▶ │  dynamic.integrate   │ ──▶ next physical_state
collision force ──▶ │   (jit + vmap-able)  │
   state, key  ──▶  └──────────────────────┘
```

---

## 2. Base API (`src/camar/dynamics/base.py`)

```python
@struct.dataclass
class PhysicalState(ABC):
    agent_pos: ArrayLike  # (num_agents, 2) — required for collisions/rendering

    @classmethod
    @abstractmethod
    def create(cls, key, agent_pos) -> "PhysicalState": ...


class BaseDynamic(ABC):
    @property
    @abstractmethod
    def action_size(self) -> int: ...

    @property
    @abstractmethod
    def dt(self) -> float: ...

    @property
    @abstractmethod
    def state_class(self) -> Type[PhysicalState]: ...

    @abstractmethod
    def integrate(
        self,
        key: ArrayLike,
        force: ArrayLike,           # (num_agents, 2) collision force, may be ignored
        physical_state: PhysicalState,
        actions: ArrayLike,          # (num_agents, action_size)
    ) -> PhysicalState: ...
```

Every concrete state must expose `agent_pos: (num_agents, 2)` so the env's collision
detection and the renderer remain dynamic-agnostic. Extra fields (`agent_vel`,
`agent_angle`, …) are dynamic-specific and accessed through the concrete state class.

### 2.1 Proposed: `inverse(state, next_state) → action`

To support trajectory replay, imitation learning, and "what action would produce this
transition?" queries, every dynamic should also implement an **inverse** method:

```python
@abstractmethod
def inverse(
    self,
    physical_state: PhysicalState,
    next_physical_state: PhysicalState,
) -> ArrayLike:                       # (num_agents, action_size), normalised to [-1, 1]
    """
    Return the action that, if passed to integrate(), reproduces ``next_physical_state``
    from ``physical_state`` (collision force = 0, no current, deterministic noise).
    Output is clipped to the dynamic's action range; transitions infeasible under the
    dynamic's action limits will be saturated at ±1 and the inverse is then approximate.
    """
```

Properties expected from `inverse`:

- **Round-trip exactness within bounds.** If `actions` is in `[-1, 1]^k` and the
  state was produced by `integrate(state, actions, force=0, current=0)`, then
  `inverse(state, next_state) ≈ actions` up to floating-point error.
- **Saturation outside bounds.** If the requested transition exceeds the action
  limits (e.g. step too large for `max_speed`), `inverse` returns the clipped action
  that produces the closest reachable next state.
- **JIT/vmap friendly.** Pure JAX, no Python branches over array values.

Existing replay code (`AISReplayPolicy`) already builds actions from differenced
trajectories by hand; `inverse` formalises and unifies that logic per dynamic.

---

## 3. Existing Dynamics

### 3.1 `DeltaPosDynamic` — kinematic position-delta with heading

File: `src/camar/dynamics/delta_pos.py`

State (`DeltaPosState`):

| field         | shape           | meaning                          |
|---------------|-----------------|----------------------------------|
| `agent_pos`   | `(N, 2)` km     | world position                   |
| `agent_angle` | `(N,)` rad      | heading; 0 = north, CW positive  |

Action: `(N, 3)` — `[Δx, Δy, Δheading]`, all in `[-1, 1]`.

Update:
```python
new_pos   = agent_pos + actions[:, :2] * max_speed
raw_angle = agent_angle + actions[:, 2] * max_angle_delta
new_angle = atan2(sin(raw_angle), cos(raw_angle))   # wrap to [-π, π]
```

Hyperparameters: `max_speed` (km / step at action = ±1), `max_angle_delta` (rad / step
at action = ±1), `dt` (seconds / step, used only for reporting `step_dt`).

Notes:
- Position is commanded directly. Collision **force is ignored** — the env still
  detects and reports collisions but the dynamic does not push agents apart.
- No velocity state. Acceleration is undefined; agents can change direction
  instantly within `max_angle_delta`.
- **Cannot couple to sea currents** in a physically meaningful way (see §5):
  the action *is* the displacement, so a current can only be added on top as a
  passive drift but ignores agent control authority.

Inverse:
```python
def inverse(self, s, s_next):
    delta_pos   = s_next.agent_pos - s.agent_pos
    delta_angle = s_next.agent_angle - s.agent_angle
    delta_angle = atan2(sin(delta_angle), cos(delta_angle))   # shortest signed arc
    a_xy   = delta_pos / max_speed
    a_head = delta_angle / max_angle_delta
    a = jnp.concatenate([a_xy, a_head[:, None]], axis=-1)
    return jnp.clip(a, -1.0, 1.0)
```

### 3.2 `HolonomicDynamic` — second-order force-controlled point mass

File: `src/camar/dynamics/holonomic.py`

State (`HolonomicState`): `agent_pos: (N, 2)`, `agent_vel: (N, 2)`.

Action: `(N, 2)` — desired acceleration in world frame, `[-1, 1]^2`.

Update (semi-implicit Euler):
```python
vel = (1 - damping) * vel
vel += (force + accel * actions) / mass * dt
# clip speed to max_speed if max_speed >= 0
pos += vel * dt
```

Hyperparameters: `accel`, `max_speed` (≥ 0 enforces clamp; negative = no clamp),
`damping ∈ [0, 1)`, `mass`, `dt`.

Notes:
- Collision force enters integration directly.
- Holonomic = no kinematic constraints; can move sideways instantly.
- **Couples cleanly with sea currents**: add current velocity to `vel` after damping
  but before position update (or model as drag force, see §5.2).

Inverse: solve for `actions` from `vel_next = (1-damping)*vel + accel*actions/mass * dt`
(force assumed 0):
```python
def inverse(self, s, s_next):
    a = ((s_next.agent_vel - (1 - damping) * s.agent_vel) * mass) / (accel * dt)
    return jnp.clip(a, -1.0, 1.0)
```

### 3.3 `DiffDriveDynamic` — unicycle / differential drive

File: `src/camar/dynamics/diffdrive.py`

State (`DiffDriveState`): `agent_pos: (N, 2)`, `agent_vel: (N, 2)`,
`agent_angle: (N, 1)` — body heading, 0 = +x (east), CCW positive.

Action: `(N, 2)` — `[linear_speed_norm, angular_speed_norm]`, mapped from `[-1, 1]`
to `[linear_speed_min, linear_speed_max]` and `[angular_speed_min, angular_speed_max]`.

Update:
```python
v_x = linear_speed * cos(angle)
v_y = linear_speed * sin(angle)
vel = [v_x, v_y] + (force / mass) * dt
pos += vel * dt
angle += angular_speed * dt   # wrap to [-π, π]
```

Notes:
- Non-holonomic: instantaneous lateral velocity = 0 in the body frame (modulo
  collision force).
- Heading frame differs from `DeltaPosDynamic` (east-zero CCW vs. north-zero CW).
  Renderers branch on the dynamic.
- **Couples with sea currents** as a body-frame disturbance: project current onto
  body axes, add to commanded `[v_x, v_y]` before integration.

Inverse: the heading update is decoupled, the linear part is the projection of
`(s_next.pos - s.pos) / dt` onto the body axis at `s.agent_angle`:
```python
def inverse(self, s, s_next):
    v   = (s_next.agent_pos - s.agent_pos) / dt
    fwd = jnp.stack([jnp.cos(s.agent_angle[..., 0]), jnp.sin(s.agent_angle[..., 0])], -1)
    linear = jnp.sum(v * fwd, axis=-1)
    angular = (s_next.agent_angle[..., 0] - s.agent_angle[..., 0]) / dt
    angular = jnp.arctan2(jnp.sin(angular * dt), jnp.cos(angular * dt)) / dt
    a_lin = (linear - linear_speed_min) / linear_speed_accel - 1.0
    a_ang = (angular - angular_speed_min) / angular_speed_accel - 1.0
    return jnp.clip(jnp.stack([a_lin, a_ang], -1), -1.0, 1.0)
```

### 3.4 `MixedDynamic` — heterogeneous fleet

File: `src/camar/dynamics/mixed.py`

Composes a list of dynamics with disjoint slices of agents
(`dynamics_batch`, `num_agents_batch`). Builds a synthetic `MixedState` dataclass
holding one sub-state per dynamic plus a flat `agent_pos` view for collisions/render.

API:
- `action_size = max(d.action_size for d in dynamics_batch)` — short actions are
  zero-padded by the env, then each dynamic slices its own width.
- All sub-dynamics must share `dt`.
- `integrate` dispatches per slice; collision `force` is sliced too.

Inverse: dispatch per slice and concatenate, padding shorter actions with zeros.

---

## 4. Action / state conventions

| dynamic              | action shape    | state extras                                             | heading frame      |
|----------------------|-----------------|----------------------------------------------------------|--------------------|
| `DeltaPosDynamic`    | `(N, 3)`        | `agent_angle: (N,)`                                      | 0 = N, CW          |
| `HolonomicDynamic`   | `(N, 2)`        | `agent_vel: (N, 2)`                                      | n/a                |
| `DiffDriveDynamic`   | `(N, 2)`        | `agent_vel: (N, 2)`, `agent_angle: (N, 1)`               | 0 = E, CCW         |
| `MixedDynamic`       | `(N, max(k_i))` | union of sub-states                                      | per sub-dynamic    |

All actions live in `[-1, 1]^k` after normalisation; the dynamic re-scales internally.

---

## 5. Sea currents

Real ships drift. CAMAR will accept an optional **sea-current field** alongside the
dynamic and feed it into `integrate(...)` as an additive velocity disturbance.

### 5.1 Field API

A current field is a pure JAX function of position and time:

```python
SeaCurrentFn = Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]
#                       (x,         y,         t)        ->  (vx, vy)   in km/s
```

- `x, y`: scalar or `(num_agents,)` km in the simulator's world frame.
- `t`: scalar — **normalised time** in `[0, 1]` over the episode (alternative: an
  integer step index `t ∈ {0, …, n_steps-1}`).
- Returns the local current velocity `(vx, vy)` with the same shape as `x, y`.

The field must be:
- **Pure** — no Python state; depends only on its inputs.
- **JIT-able / vmap-able** — works under `jax.jit` and `jax.vmap` over agents.
- **Smooth enough** — interpolation onto irregular agent positions; `jax.scipy`,
  precomputed grid + bilinear lookup, or analytical (e.g. tidal sinusoids).

A reference implementation will provide:

```python
def constant_current(vx: float, vy: float) -> SeaCurrentFn: ...
def gridded_current(grid_x, grid_y, vx_field, vy_field, t_axis) -> SeaCurrentFn: ...
def tidal_sinusoid(amplitude, period_steps, direction_rad) -> SeaCurrentFn: ...
def zero_current() -> SeaCurrentFn:           # default; jits to a no-op
    return lambda x, y, t: (jnp.zeros_like(x), jnp.zeros_like(y))
```

### 5.2 How dynamics consume currents

The `integrate` signature gains an optional `current_fn` (or, equivalently, a
precomputed `(vx, vy)` tensor at the agent positions and current step):

```python
def integrate(self, key, force, physical_state, actions, current=None):
    ...
```

Per-dynamic semantics:

| dynamic              | supports current | how                                                                                                |
|----------------------|------------------|----------------------------------------------------------------------------------------------------|
| `DeltaPosDynamic`    | **No**           | Action *is* displacement; current cannot be added without breaking the action contract.            |
| `HolonomicDynamic`   | Yes              | `vel += current * dt` after damping/accel update, before `max_speed` clamp and position update.    |
| `DiffDriveDynamic`   | Yes              | After computing body-frame velocity `[v_x, v_y]`, add `current` (world frame) before `pos += vel*dt`. Heading not affected. |
| `MixedDynamic`       | Mixed            | Forward to each sub-dynamic; sub-dynamics that opt out (DeltaPos) ignore the field.                |

Dynamics that opt out advertise this via a class attribute:

```python
class BaseDynamic:
    supports_current: bool = False
```

The environment reads `dynamic.supports_current`; if a current field was provided
but the dynamic does not support it, the env raises at construction time so the
mismatch surfaces early (instead of a silent zero current).

### 5.3 Time normalisation

Two options are supported:

1. **Normalised float `t ∈ [0, 1]`** — simplest for analytical fields. The env
   computes `t = step_index / max(1, n_steps - 1)`.
2. **Integer step index** — preferred when current data is sampled per simulator
   step (e.g. AIS-aligned tidal model). The env passes `step_index: int32`.

Default is the normalised float; the field author can wrap in
`jax.lax.convert_element_type` if they need an int. The choice is fixed at env
construction (`current_time_mode="float" | "step"`).

### 5.4 Rendering

The renderer can optionally sample the current field on a coarse grid and overlay
arrows or a colour field. Same `SeaCurrentFn` signature, no separate API.

---

## 6. Open questions for ship-specific dynamics

The next round of dynamics will be ship-specific (Nomoto first-order steering,
3-DoF surge/sway/yaw with rudder + thrust, …). Outstanding decisions:

- **Action spaces.** Do we keep the `[-1, 1]^k` contract, or expose physical
  units (rudder angle in rad, thrust in kN)? Recommend keeping the normalised
  contract and documenting per-dynamic ranges.
- **Wind / waves.** Same field shape as currents, or a unified "environmental
  force" `(x, y, t) → (fx, fy, mz)` returning a force + yaw moment?
- **Inverse for hydrodynamic models.** Closed-form inverse may not exist; allow
  numerical inverse (single Newton step) as a fallback.
- **Stochasticity.** `key` is already in `integrate`; reserve it for process
  noise (sensor / actuator noise belongs in env, not dynamic).

These will be resolved as the per-ship-type dynamics list is finalised.
