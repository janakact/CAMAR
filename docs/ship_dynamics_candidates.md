# Candidate Ship Dynamics Models

Survey of ship transition-dynamics models used in research, suitable for adding to
CAMAR's `BaseDynamic` registry. Ordered ship-specific first (MMG, Fossen 3-DoF,
Abkowitz, Nomoto, Norrbin), then generic kinematic and data-driven fallbacks.

Each entry lists: state, action, equations, where it shines, where it fails, sea-
current coupling, and inverse-dynamics tractability.

See `transition_dynamics.md` for the existing API contract.

---

## 1. Ship-specific hydrodynamic models

### 1.1 MMG (Manoeuvring Modelling Group) — **primary target**
- **State.** 3-DoF `η = [x, y, ψ]`, `ν = [u, v, r]`.
- **Action.** Rudder angle `δ`, propeller revolutions `n` (or `n_norm ∈ [-1, 1]`).
- **Equation.** Forces split by component:
  - Hull: `X_H, Y_H, N_H` from polynomial expansions in `(u, v, r)`.
  - Propeller: `X_P = (1 − t_P) ρ n² D_P⁴ K_T(J_P)`.
  - Rudder: `X_R, Y_R, N_R` from `δ`, rudder normal force `F_N(α_R, U_R)`.
  - Plus hull-propeller-rudder interaction coefficients (`a_H`, `x_H`, `t_R`, ...).
  - `M ν̇ + C(ν) ν = τ_H + τ_P + τ_R + τ_env`.
- **Use.** De-facto industry standard for ship manoeuvrability prediction (IMO,
  shipyards). Modular: swap rudder / propeller without re-identifying the hull.
- **Limits.** Identification non-trivial; many empirical interaction terms.
- **Currents.** Standard: relative velocity `ν_r = ν − R(ψ)ᵀ V_c` in hull forces.
- **Inverse.** Non-affine in actions (propeller curve, rudder-angle nonlinearity);
  numerical only (single Newton step is usually enough).
- **References.**
  - Yasukawa & Yoshimura (2015), *Introduction of MMG standard method for ship
    manoeuvring predictions*, J. Marine Science and Technology.
  - Springer chapter: <https://link.springer.com/content/pdf/10.1007/978-981-97-0625-9_10.pdf>
  - Real-scale parameter fine-tuning: <https://arxiv.org/pdf/2312.04224>
  - System ID review: <https://link.springer.com/article/10.1007/s11804-025-00681-w>
  - Deep + shallow water variant: <https://www.sciencedirect.com/science/article/pii/S0029801824002646>

### 1.2 Fossen 3-DoF manoeuvring (MSS toolbox)
- **State.** `η = [x, y, ψ]`, `ν = [u, v, r]`.
- **Action.** Generalised force `τ = [X, Y, N]` (or rudder-thrust mapped onto `τ`).
- **Equation.**
  ```
  η̇ = R(ψ) ν
  M ν̇ + C(ν) ν + D(ν) ν = τ + τ_env
  ```
  `M` = rigid-body + added mass; `C(ν)` Coriolis-centripetal; `D(ν)` linear +
  quadratic damping.
- **Use.** Most-cited template in marine-RL papers; clean JAX implementation
  (constant `M⁻¹`, vectorisable `C, D`). Sweet spot for autonomy work.
- **Currents.** First-class: relative velocity `ν_r = ν − ν_c` enters `C(ν_r)`
  and `D(ν_r)`.
- **Inverse.** Closed-form per step:
  `τ = M ν̇ + C(ν) ν + D(ν) ν`, with `ν̇ ≈ (ν_next − ν) / dt`.
- **References.**
  - Fossen (2021), *Handbook of Marine Craft Hydrodynamics and Motion Control*,
    2nd ed., Wiley. <https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>
  - Fossen marine craft model page: <https://fossen.biz/html/marineCraftModel.html>
  - MSS toolbox: <https://github.com/cybergalactic/MSS>
  - CyberShip II identified parameters:
    <https://www.sciencedirect.com/science/article/pii/S1474667017317329>
  - Skjetne et al. (2004), *A Nonlinear Ship Manoeuvering Model: ID and adaptive
    control with experiments*. <https://www.mic-journal.no/PDF/2004/MIC-2004-1-1.pdf>

### 1.3 Abkowitz whole-ship model
- **State.** Full 3-DoF `(η, ν)`.
- **Action.** `δ`, propeller revs `n` (or `τ`).
- **Equation.** Truncated Taylor series in `(u, v, r, δ)` up to 3rd order;
  hundreds of hydrodynamic derivatives bundled into `X(·), Y(·), N(·)`.
- **Use.** High-fidelity simulation when captive-model or CFD data available.
- **Limits.** ~70+ coefficients per ship; data-hungry; no separable hull /
  propeller / rudder interaction (everything is implicit in the polynomial).
- **Currents.** Via `τ_env` or relative-velocity substitution.
- **Inverse.** Numerical (cubic in `δ` and `(u, v, r)`).
- **References.**
  - Abkowitz (1964), *Lectures on Ship Hydrodynamics — Steering and
    Manoeuvrability*, Hydro- and Aerodynamics Lab Report Hy-5.
  - ID with ε-SVR: <https://www.researchgate.net/publication/241105238_Identification_of_Abkowitz_Model_for_Ship_Manoeuvring_Motion_Using_e_-Support_Vector_Regression>
  - Modern comparison: <https://www.tandfonline.com/doi/full/10.1080/17445302.2022.2067409>

### 1.4 Davidson–Schiff linear manoeuvring
- **State.** `(x, y, ψ, u, v, r)` with `u ≈ U₀` constant.
- **Action.** Rudder `δ` (+ optional thrust).
- **Equation.** Linearised sway-yaw subsystem:
  `(m − Y_v̇) v̇ + (m U − Y_r) r − Y_v v = Y_δ δ`,
  `(I_z − N_ṙ) ṙ − N_v v − N_r r = N_δ δ`.
- **Use.** Foundational; basis for Nomoto reduction. Good linear-control
  baseline / sanity check.
- **Currents.** Substitute relative velocity for hull forces.
- **Inverse.** Linear algebra solve for `δ` per step.
- **Reference.**
  - Davidson & Schiff (1946), *Turning and course-keeping qualities*,
    Stevens Institute of Technology. (Foundational; not online — cited via
    Fossen handbook and <https://arxiv.org/html/2502.18696v1>.)

---

## 2. Ship-specific steering-response models (1-DoF yaw)

Reduced models — useful for heading autopilots, course-keeping RL, and as
lightweight surrogates inside large multi-agent simulations.

### 2.1 Nomoto 1st-order (KT model)
- **State.** `(x, y, ψ, r)`. Forward speed `U` parameter (or slow extra state).
- **Action.** Rudder angle `δ ∈ [δ_min, δ_max]`.
- **Equation.**
  `T ṙ + r = K δ`, `ψ̇ = r`, `ẋ = U cos ψ`, `ẏ = U sin ψ`.
- **Use.** Heading autopilot, course-keeping. Two parameters `(K, T)` identifiable
  from a single zig-zag test.
- **Limits.** Linear; ignores sway, surge dynamics, large-rudder nonlinearity.
- **Currents.** Add to `(ẋ, ẏ)` directly.
- **Inverse.** Closed-form: `δ = (T ṙ + r) / K`.
- **References.**
  - Nomoto, Taguchi, Honda, Hirano (1957), *On the steering qualities of ships*.
  - Sutulo & Guedes Soares, *Fundamental Properties of Linear Ship Steering
    Dynamic Models*. <https://www.researchgate.net/publication/265324077_Fundamental_Properties_of_Linear_Ship_Steering_Dynamic_Models>
  - Autopilot design: <https://www.researchgate.net/publication/277497099_Ships_Steering_Autopilot_Design_by_Nomoto_Model>

### 2.2 Nomoto 2nd-order (Nomoto–Bech)
- **State.** `(x, y, ψ, r, v)` (or `ψ, r, ṙ`).
- **Action.** Rudder `δ`.
- **Equation.** `T₁ T₂ r̈ + (T₁ + T₂) ṙ + r = K (δ + T₃ δ̇)`. Near-cancelled
  zero/pole makes this ill-conditioned for identification.
- **Use.** Captures sway-yaw coupling overshoot; better fit than 1st-order for
  large ships at low speed.
- **Inverse.** Closed-form (numerical safer).
- **References.** Sutulo & Guedes Soares (above);
  <https://shipjournal.co/index.php/sst/article/view/160>

### 2.3 Norrbin nonlinear steering
- **State.** `(x, y, ψ, r)`.
- **Action.** Rudder `δ`.
- **Equation.** `T ṙ + n₃ r³ + n₂ r² + n₁ r + n₀ = K δ`. Polynomial term
  captures course instability of bulk carriers / tankers.
- **Use.** Course-keeping under heavy seas, large vessels.
- **Currents.** Same as Nomoto-1.
- **Inverse.** Polynomial in `δ` (trivial); cubic in `r` for state-prediction.
- **References.**
  - Norrbin (1963), *On the design and analysis of zig-zag tests*.
  - Application example: <https://www.mdpi.com/2077-1312/13/3/534>
  - Course-keeping under disturbance: <https://www.sciencedirect.com/science/article/abs/pii/S0029801821016796>

---

## 3. 6-DoF ship models (seakeeping)

### 3.1 Fossen 6-DoF
- **State.** `η = [x, y, z, φ, θ, ψ]`, `ν = [u, v, w, p, q, r]`.
- **Action.** Wrench `τ ∈ R⁶`.
- **Equation.** Same form as 3-DoF Fossen with full `M, C, D`. Wave excitation
  modelled separately as `τ_wave`.
- **Use.** When roll, pitch, heave matter — heavy-weather, parametric roll,
  green-water, stabiliser-fin control.
- **Limits.** Expensive; needs wave model; rarely used in traffic / collision-
  avoidance RL.
- **Currents.** Same relative-velocity treatment.
- **Inverse.** Closed-form in rigid-body part; wave loads exogenous.
- **References.**
  - Fossen handbook (above).
  - 6-DoF ID with wave loads:
    <https://www.researchgate.net/publication/362750653_Data-driven_simultaneous_identification_of_the_6DOF_dynamic_model_and_wave_load_for_a_ship_in_waves>
  - DL prediction: <https://journals.sagepub.com/doi/full/10.1177/14750902231157852>
  - Turning in waves: <https://www.nature.com/articles/s41598-026-46427-8>

---

## 4. Underactuated USV variants

Same equations as §1.2 (Fossen 3-DoF) but with control authority limited to
`τ = [X, 0, N]` (surge thrust + yaw moment, no direct sway). Most autonomous-
surface-vessel literature operates in this regime.

- **State.** `(η, ν)` same as Fossen 3-DoF.
- **Action.** `(τ_X, τ_N)` — or `(n, δ)` if rudder-screw, or
  `(n_port, n_stbd)` if differential-thruster.
- **References.**
  - Cui et al., *Underactuated USV path following — cascade method*, Sci.
    Reports (2022). <https://www.nature.com/articles/s41598-022-05456-9>
  - Rudderless double-thrusters: <https://pmc.ncbi.nlm.nih.gov/articles/PMC6539673/>
  - USV ID with experimental results:
    <https://www.sciencedirect.com/science/article/abs/pii/S0029801821013688>
  - Path-following with current disturbance:
    <https://journals.sagepub.com/doi/full/10.1177/1729881419871807>
  - USVs-Sim platform: <https://onlinelibrary.wiley.com/doi/10.1002/cpe.6567>

---

## 5. Generic / kinematic fallbacks (non-ship-specific)

Useful for AIS replay, multi-target tracking, and toy RL where hydrodynamics are
overkill.

### 5.1 Constant Velocity (CV)
- **State.** `(x, y, vx, vy)`.
- **Action.** None or `(vx, vy)` setpoint.
- **Equation.** `pos += vel · dt`.
- **Currents.** `vel_eff = vel + current`.
- **Inverse.** Closed-form.
- **Reference.** Tam, Bucknall, Greig (2009), *A Simplified Simulation Model of
  Ship Navigation for Safety and Collision Avoidance in Heavy Traffic Areas*.
  <https://www.cambridge.org/core/journals/journal-of-navigation/article/abs/simplified-simulation-model-of-ship-navigation-for-safety-and-collision-avoidance-in-heavy-traffic-areas/D8671045BDAC618F5330FE4E44BB4848>

### 5.2 Constant Turn-Rate and Velocity (CTRV)
- **State.** `(x, y, ψ, v, ω)`.
- **Equation.** Arc step:
  `ψ += ω dt`,
  `x += v/ω · (sin ψ_new − sin ψ)`,
  `y += v/ω · (cos ψ − cos ψ_new)` (linearised at small `ω`).
- **Use.** AIS interpolation, MOT filters.
- **Inverse.** Closed-form.

### 5.3 Unicycle / Bicycle (current `DiffDriveDynamic`)
- Already implemented. See `transition_dynamics.md` §3.3.

---

## 6. Data-driven / hybrid models

### 6.1 Fossen base + NN / GP residual
- Fossen 3-DoF + learned residual `f_θ(η, ν, τ)` correcting un-modelled effects.
- Strong sim-to-real on a specific hull.
- **References.**
  - <https://html.rhhz.net/jmsa/html/20160407.htm>
  - <https://www.mdpi.com/2073-8994/13/10/1956>
  - <https://arxiv.org/html/2502.18696v1>
  - <https://link.springer.com/article/10.1007/s00773-024-01045-9>

### 6.2 Pure NN / Koopman surrogates
- Sequence model (LSTM, Transformer, Koopman operator) maps
  `(state, action) → state_next`. No physics prior.
- Useful as a learned simulator twin; lossy inverse (one optimisation step).
- **References.**
  - <https://www.sciencedirect.com/science/article/abs/pii/S0029801823016396>
  - <https://journals.sagepub.com/doi/full/10.1177/14750902231157852>

---

## 7. Comparison table

| #   | Model              | State dim | Action dim | Coefs to ID | Inverse   | Currents | RL fit                     |
|-----|--------------------|-----------|------------|-------------|-----------|----------|----------------------------|
| 1.1 | MMG                | 6         | 2 (δ, n)   | ~30 modular | numerical | yes      | industry standard          |
| 1.2 | Fossen 3-DoF       | 6         | 3 (τ)      | ~15         | closed    | yes      | marine-RL default          |
| 1.3 | Abkowitz           | 6         | 2          | ~70         | numerical | yes      | high fidelity              |
| 1.4 | Davidson–Schiff    | 6         | 1–2        | ~10         | closed    | yes      | linear-control baseline    |
| 2.1 | Nomoto-1           | 4         | 1          | 2 (K, T)    | closed    | yes      | heading control            |
| 2.2 | Nomoto-2           | 5         | 1          | 4–5         | closed    | yes      | heading control            |
| 2.3 | Norrbin            | 4         | 1          | 4 (poly)    | closed    | yes      | tankers, big ships         |
| 3.1 | Fossen 6-DoF       | 12        | 6          | ~30 + waves | closed*   | yes      | seakeeping RL              |
| 4   | Underactuated USV  | 6         | 2          | ~15         | closed    | yes      | autonomous surface vessels |
| 5.1 | Constant Velocity  | 4         | 0 / 2      | 0           | closed    | trivial  | replay only                |
| 5.2 | CTRV               | 5         | 0 / 2      | 0           | closed    | trivial  | replay, MOT                |
| 5.3 | Unicycle           | 5         | 2          | few         | closed    | yes      | toy RL                     |
| 6.1 | Hybrid Fossen + NN | 6–12      | 2–6        | base + NN   | numerical | yes      | sim-to-real                |
| 6.2 | Pure NN / Koopman  | latent    | 2–6        | NN only     | numerical | implicit | learned twin               |

`closed*` = closed-form in the rigid-body part; exogenous wave loads aside.

---

## 8. Recommendation for CAMAR roadmap

Phase order, ship-specific first. Each phase can land independently and reuses
the sea-current `SeaCurrentFn` from `transition_dynamics.md` §5.

1. **`NomotoFirstOrderDynamic`** — lightest ship-specific dynamic; closed-form
   inverse; smallest delta over existing `DiffDriveDynamic`.
2. **`Fossen3DOFDynamic`** — full surge/sway/yaw with `τ` actions; the marine-RL
   default; natural current support via relative velocity; closed-form inverse.
3. **`MMGDynamic`** — component-wise `(hull / propeller / rudder)` action set
   (rudder angle + RPM); reuses the Fossen 3-DoF integrator with a different
   force-assembly module.
4. **`NorrbinDynamic`** — drop-in upgrade over Nomoto-1 for course-unstable big
   ships.
5. **`UnderactuatedUSVDynamic`** — Fossen 3-DoF restricted to `(τ_X, τ_N)` for
   USV / autonomous-surface-vessel scenarios.
6. **(Optional) `AbkowitzDynamic`** — gated on an identified coefficient set per
   hull.

6-DoF and pure-NN surrogates remain out of scope until seakeeping / sim-to-real
becomes a goal.

---

## 9. State additions per dynamic (delta over current `(x, y, ψ)`)

Current CAMAR `DeltaPosState` already stores `agent_pos = (x, y)` and
`agent_angle = ψ` (rad, 0 = north, CW). This section lists **only what each
candidate dynamic needs on top** — extra physical-state fields, plus actuator
state when relevant.

### Ship-specific

#### 1.1 MMG
Extra state:
- `u` — surge velocity (body x), m/s.
- `v` — sway velocity (body y), m/s.
- `r` — yaw rate, rad/s.
- `n` — propeller revolutions (rev/s), if rev dynamics modelled. If commanded
  directly each step, store last command instead.
- `δ` — rudder angle (rad), if rudder servo dynamics modelled
  (`δ̇ = (δ_cmd − δ) / T_δ`, saturated to `[δ_min, δ_max]`). If rudder
  responds instantly, store only the command for inverse-dynamics.

Optional but recommended:
- `U = sqrt(u² + v²)` — cached scalar speed.
- `β = atan2(−v, u)` — drift angle, useful for hull-force lookups.

Total extra (minimal modelled-actuator variant): **5 scalars per agent**
(`u, v, r, n, δ`).

#### 1.2 Fossen 3-DoF
Extra state:
- `u, v, r` — body-frame velocities (same as MMG).

No actuator state if `τ = [X, Y, N]` is commanded directly. Add `δ, n` only if
you map `τ` through an explicit rudder/propeller model.

Total extra: **3 scalars per agent**.

#### 1.3 Abkowitz
Same as Fossen 3-DoF: `u, v, r`, plus optionally `δ, n` for actuator dynamics.
Total: **3–5 scalars per agent**.

#### 1.4 Davidson–Schiff
Same as Fossen 3-DoF: `u, v, r` (with `u ≈ U₀` clamped to a constant if you
drop surge from the state). Minimal variant: only `v, r` if `u = U₀` constant.
Total: **2–3 scalars per agent**.

### Steering-response

#### 2.1 Nomoto-1
Extra state:
- `r` — yaw rate, rad/s.
- `U` — forward speed (m/s). Either a constant parameter (no state) or a slow
  surge state with its own first-order lag.

Optional actuator: `δ` (rudder), with `δ̇ = (δ_cmd − δ) / T_δ`.
Total extra: **1 scalar** (just `r`, if `U` and `δ` are parameters).

#### 2.2 Nomoto-2
Extra state:
- `r` and either `ṙ` or `v` (sway), depending on which 2nd-order form you pick.
- `U` (same caveats as Nomoto-1).

Optional: `δ` actuator state.
Total extra: **2 scalars** (`r, ṙ` or `r, v`).

#### 2.3 Norrbin
Same as Nomoto-1: **1 scalar** (`r`). Nonlinearity is in the dynamics, not the
state.

### 6-DoF / underactuated

#### 3.1 Fossen 6-DoF
Extra pose: `z, φ, θ` (heave, roll, pitch).
Extra body velocities: `w, p, q` (plus existing-from-3DoF `u, v, r`).
Optional wave state: a small spectral filter state per agent (e.g. 2nd-order
shaping filter for `τ_wave`) — depends on wave model.
Total extra: **9 scalars** for rigid-body (`z, φ, θ, u, v, w, p, q, r`),
plus wave-filter state if used.

#### 4 Underactuated USV
Identical state to Fossen 3-DoF (`u, v, r`); only the action set is restricted.
Total extra: **3 scalars**.

### Generic kinematic fallbacks

#### 5.1 Constant Velocity
Extra state: `vx, vy` (or `U, ψ` if you reuse `ψ`).
Total extra: **2 scalars**.

#### 5.2 CTRV
Extra state: `v` (speed scalar), `ω` (turn rate).
Total extra: **2 scalars**.

#### 5.3 Unicycle (current `DiffDriveDynamic`)
Already implemented; extras are `agent_vel = (vx, vy)` per existing
`DiffDriveState`. No new state.

### Data-driven

#### 6.1 Fossen + NN residual
Base state same as §1.2 (`u, v, r`). NN residual is stateless if it's a
feed-forward correction; add a recurrent latent `h ∈ R^k` if you use an
RNN/LSTM residual.

#### 6.2 Pure NN / Koopman
Latent state `z ∈ R^k` (size set by the model). `(x, y, ψ)` may be a decoded
view of `z` or carried alongside. Total extra: **k scalars** (k typically
16–64).

---

## 10. Suggested unified state schema

To keep `PhysicalState` subclasses interoperable (mixed fleets), pre-allocate
one canonical superset and let each dynamic update only the fields it owns.
Strawman:

```python
@struct.dataclass
class ShipState(PhysicalState):
    agent_pos:    ArrayLike   # (N, 2)   x, y  [km]
    agent_angle:  ArrayLike   # (N,)     ψ      [rad, 0 = north, CW]
    agent_vel:    ArrayLike   # (N, 2)   u, v  body-frame [m/s] — zeros for kinematic
    agent_r:      ArrayLike   # (N,)     yaw rate [rad/s]
    rudder:       ArrayLike   # (N,)     δ      [rad]  — zeros if no actuator dyn
    prop_rev:     ArrayLike   # (N,)     n      [rev/s] — zeros if no actuator dyn
```

Per-dynamic ownership:

| Dynamic            | `agent_vel` | `agent_r` | `rudder` | `prop_rev` |
|--------------------|-------------|-----------|----------|------------|
| DeltaPos (current) | —           | —         | —        | —          |
| Nomoto-1           | —           | ✓         | opt.     | —          |
| Nomoto-2           | sway only   | ✓ + `ṙ`*  | opt.     | —          |
| Norrbin            | —           | ✓         | opt.     | —          |
| Davidson–Schiff    | ✓           | ✓         | ✓        | opt.       |
| Fossen 3-DoF       | ✓           | ✓         | opt.     | opt.       |
| MMG                | ✓           | ✓         | ✓        | ✓          |
| Abkowitz           | ✓           | ✓         | ✓        | ✓          |
| Fossen 6-DoF       | ✓ + `w`     | ✓ + `p,q` | ✓        | ✓          |
| Underactuated USV  | ✓           | ✓         | opt.     | opt.       |
| CV / CTRV          | ✓           | (CTRV ✓)  | —        | —          |

`*` Nomoto-2's `ṙ` needs a separate slot; either widen `agent_r` to `(N, 2)`
or add a dedicated `agent_r_dot` field.

This schema means: **every ship-specific dynamic needs `(u, v, r)` on top of
`(x, y, ψ)`** — that's the irreducible minimum. MMG / Abkowitz additionally
need `(δ, n)` when modelling actuator lag. Everything else is opt-in.

---

## Sources

- [Review of System Identification for Manoeuvring Modelling (Springer, 2025)](https://link.springer.com/article/10.1007/s11804-025-00681-w)
- [Comparison of ship manoeuvrability models (T&F, 2022)](https://www.tandfonline.com/doi/full/10.1080/17445302.2022.2067409)
- [Math-data integrated prediction model (Ocean Eng., 2023)](https://www.sciencedirect.com/science/article/abs/pii/S0029801823016396)
- [MMG fine-tuning (arXiv 2312.04224)](https://arxiv.org/pdf/2312.04224)
- [The MMG Model (Springer chapter)](https://link.springer.com/content/pdf/10.1007/978-981-97-0625-9_10.pdf)
- [Abkowitz ID with ε-SVR (ResearchGate)](https://www.researchgate.net/publication/241105238_Identification_of_Abkowitz_Model_for_Ship_Manoeuvring_Motion_Using_e_-Support_Vector_Regression)
- [Ship Manoeuvring with Neural Networks](https://html.rhhz.net/jmsa/html/20160407.htm)
- [Maneuvering in deep and shallow waters (Ocean Eng., 2024)](https://www.sciencedirect.com/science/article/pii/S0029801824002646)
- [Fossen Marine Craft Model page](https://fossen.biz/html/marineCraftModel.html)
- [Handbook of Marine Craft Hydrodynamics and Motion Control (Wiley, 2021)](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138)
- [MSS Marine Systems Simulator (GitHub)](https://github.com/cybergalactic/MSS)
- [FossenHandbook companion (GitHub)](https://github.com/cybergalactic/FossenHandbook)
- [CyberShip II modeling, ID, adaptive maneuvering (IFAC / ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1474667017317329)
- [A Nonlinear Ship Manoeuvering Model (MIC, 2004)](https://www.mic-journal.no/PDF/2004/MIC-2004-1-1.pdf)
- [Turning in waves, 6-DoF (Sci. Reports, 2026)](https://www.nature.com/articles/s41598-026-46427-8)
- [Data-driven 6DOF ID + wave load (ResearchGate)](https://www.researchgate.net/publication/362750653_Data-driven_simultaneous_identification_of_the_6DOF_dynamic_model_and_wave_load_for_a_ship_in_waves)
- [DL prediction of 6-DoF ship motions (Sage, 2023)](https://journals.sagepub.com/doi/full/10.1177/14750902231157852)
- [Fundamental Properties of Linear Ship Steering Dynamic Models](https://www.researchgate.net/publication/265324077_Fundamental_Properties_of_Linear_Ship_Steering_Dynamic_Models)
- [Ships Steering Autopilot Design by Nomoto Model](https://www.researchgate.net/publication/277497099_Ships_Steering_Autopilot_Design_by_Nomoto_Model)
- [First and Second Order Nomoto — Fluvial Support Patrol ID](https://shipjournal.co/index.php/sst/article/view/160)
- [Nonlinear Course-Keeping (Norrbin, MDPI 2025)](https://www.mdpi.com/2077-1312/13/3/534)
- [Course-keeping under disturbance (Ocean Eng., 2021)](https://www.sciencedirect.com/science/article/abs/pii/S0029801821016796)
- [Underactuated USV path-following — cascade method (Sci. Reports, 2022)](https://www.nature.com/articles/s41598-022-05456-9)
- [USV with Rudderless Double Thrusters (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6539673/)
- [Modeling, parameter ID, guidance and control of a USV (Ocean Eng., 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0029801821013688)
- [Path-following with current disturbance (IJARS, 2019)](https://journals.sagepub.com/doi/full/10.1177/1729881419871807)
- [USVs-Sim simulation platform (Wiley, 2022)](https://onlinelibrary.wiley.com/doi/10.1002/cpe.6567)
- [Simplified Ship Navigation Model (Cambridge)](https://www.cambridge.org/core/journals/journal-of-navigation/article/abs/simplified-simulation-model-of-ship-navigation-for-safety-and-collision-avoidance-in-heavy-traffic-areas/D8671045BDAC618F5330FE4E44BB4848)
- [Interpretable Data-Driven Ship Dynamics (arXiv 2502.18696)](https://arxiv.org/html/2502.18696v1)
- [Ship Dynamics ID via Sparse GP Regression (MDPI Symmetry)](https://www.mdpi.com/2073-8994/13/10/1956)
- [Discovering ship maneuvering models from data (JMST)](https://link.springer.com/article/10.1007/s00773-024-01045-9)
