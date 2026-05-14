# Ship Dynamics — Plain-Language Reference

Companion to `ship_dynamics_candidates.md`. No equations. Each entry: short
description of what is being modelled, the meaning of every symbol used, a
textual description of how the pieces interact, how sea-current information is
consumed (if at all), and sources.

A handful of symbols recur across models:

- **x, y** — world-frame position (north, east) in km or m.
- **ψ (psi)** — heading angle: which way the bow points.
- **u** — surge velocity: forward speed along the ship's own length axis.
- **v** — sway velocity: sideways speed across the ship's beam axis.
- **r** — yaw rate: how fast the heading is changing.
- **U** — speed over ground (scalar): magnitude of the velocity vector.
- **β (beta)** — drift angle: angle between the bow and the actual direction of motion.
- **δ (delta)** — rudder angle: how far the rudder is turned.
- **n** — propeller revolutions: how fast the propeller is spinning.
- **τ (tau)** — generalised force / moment applied to the hull.
- **V_c** — sea-current velocity vector at the ship's location.
- **ν_r** — relative velocity of the ship through the water (ground speed minus current).

---

## 1. Ship-specific hydrodynamic models

### 1.1 MMG (Manoeuvring Modelling Group) — primary target

Models a real ship as **three interacting components**: hull, propeller, rudder.
Each component is a separate force / moment generator, and the model adds them
up to push the ship around.

**Symbols.**
- **x, y, ψ** — pose.
- **u, v, r** — surge / sway / yaw velocities of the hull.
- **δ** — rudder angle commanded to the steering gear.
- **n** — propeller revolutions per second.
- **V_c** — sea-current vector at the ship.
- **ν_r** — hull's velocity through the water (ground motion minus current).

**Interaction.** The propeller spinning at *n* pushes water backward and shoves
the ship forward; the speed of that push depends on how fast water is already
flowing past the propeller. The rudder sits in the propeller's wake, so when
the rudder is turned to *δ*, it deflects accelerated water and produces a
sideways force at the stern, which yaws and sways the ship. The hull resists
all of this through hydrodynamic drag that depends on its own surge, sway and
yaw motion. The three modules pass interaction coefficients between each other
(rudder-in-propeller-wake, hull-rudder lift augmentation, propeller wake
fraction), so changing one component changes the others. Engineers can swap
the rudder or propeller model without re-identifying the hull, which is why
shipyards and IMO use it.

**Action space.** Two dimensions:
- **a₀ — rudder command** *δ_cmd*: how far to turn the rudder, normalised to
  `[-1, 1]` and re-scaled to the rudder's physical range (typically ±35°).
  Negative = port (left), positive = starboard (right).
- **a₁ — propeller throttle** *n_cmd*: how fast to spin the propeller,
  normalised to `[-1, 1]` and mapped to a revolutions-per-second range.
  Negative = astern (reverse), positive = ahead.

**Sea currents.** **Yes — first class.** The hull's force generator does not
care about ground velocity; it cares about velocity through the water. So
every place the model uses *(u, v, r)* internally, it really uses *ν_r* —
ground velocity minus the local current vector *V_c*. Currents push the ship
sideways even when the rudder is centred, and they also change the drift angle
the hull sees, which changes hull damping forces. The simulator must sample
*V_c* at the ship's *(x, y)* each step and pass it into the hull-force module.

**Sources.**
- Yasukawa & Yoshimura (2015), *Introduction of MMG standard method for ship manoeuvring predictions*.
- <https://link.springer.com/content/pdf/10.1007/978-981-97-0625-9_10.pdf>
- <https://arxiv.org/pdf/2312.04224>
- <https://link.springer.com/article/10.1007/s11804-025-00681-w>
- <https://www.sciencedirect.com/science/article/pii/S0029801824002646>

---

### 1.2 Fossen 3-DoF (MSS toolbox)

Models the ship as **one rigid body in the horizontal plane** driven by a
single lumped force / moment vector. Does not split forces by component — they
are all packed into one generalised wrench.

**Symbols.**
- **x, y, ψ** — pose.
- **u, v, r** — surge / sway / yaw velocities.
- **τ** — control wrench (surge force, sway force, yaw moment).
- **τ_env** — environmental wrench (wind, waves, current).
- **V_c** — sea-current vector.
- **ν_r** — relative velocity through the water.

**Interaction.** The control wrench *τ* is the only thing the agent gets to
choose. It accelerates the hull along its body axes, but acceleration is
resisted by the hull's inertia (including added mass — water dragged along
with the ship), by Coriolis-like coupling between surge / sway / yaw, and by
hydrodynamic damping that grows with speed. Heading changes happen indirectly:
yaw moment changes *r*, which integrates into *ψ*, which rotates the body
frame so that the next *τ* points in a different world direction. Cleanly
separates "what the controller asks for" (*τ*) from "what the water does"
(damping, Coriolis, added mass).

**Action space.** Three dimensions (the components of *τ*):
- **a₀ — surge force** *τ_X*: how hard to push forward / backward along the
  ship's length. Normalised to `[-1, 1]`, re-scaled to a max thrust (N).
- **a₁ — sway force** *τ_Y*: sideways push across the beam. Available only on
  vessels with tunnel thrusters or azimuth pods. Normalised to `[-1, 1]`,
  re-scaled to a max side-force.
- **a₂ — yaw moment** *τ_N*: turning torque about the vertical axis.
  Normalised to `[-1, 1]`, re-scaled to a max moment (N·m).

**Sea currents.** **Yes — first class.** All damping and Coriolis terms use
*ν_r*, the velocity of the hull relative to the water. The simulator subtracts
*V_c* from the body-frame velocity before feeding it into damping and
Coriolis, and adds *V_c* back into the world-frame velocity used for position
integration. With no current, *ν_r* = body velocity and the model collapses
to the standard form.

**Sources.**
- <https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>
- <https://fossen.biz/html/marineCraftModel.html>
- <https://github.com/cybergalactic/MSS>
- <https://www.sciencedirect.com/science/article/pii/S1474667017317329>
- <https://www.mic-journal.no/PDF/2004/MIC-2004-1-1.pdf>

---

### 1.3 Abkowitz whole-ship

Models the ship as **one rigid body** like Fossen 3-DoF, but builds the
hydrodynamic forces as a big polynomial in the ship's own motion and the
rudder angle, with no separation between hull, propeller and rudder.

**Symbols.**
- **x, y, ψ** — pose.
- **u, v, r** — surge / sway / yaw velocities.
- **δ** — rudder angle.
- **n** — propeller revolutions (when separately modelled).
- **V_c, ν_r** — current vector and relative velocity through water.

**Interaction.** Every force on the ship is treated as a smooth function of
*(u, v, r, δ)*. The polynomial coefficients implicitly bundle hull damping,
rudder lift, propeller thrust and all their interactions. Changing the rudder
both yaws the ship and changes the surge drag (because the polynomial has
cross terms). This is more faithful to wind-tunnel-style test data, but it
also means re-identifying ~70 coefficients per hull. Once identified, runtime
is cheap — just polynomial evaluation.

**Action space.** Two dimensions:
- **a₀ — rudder angle** *δ*: how far to turn the rudder. Normalised to
  `[-1, 1]`, re-scaled to ±δ_max (typically ±35°). Negative = port,
  positive = starboard.
- **a₁ — propeller revolutions** *n*: how fast to spin the propeller.
  Normalised to `[-1, 1]`, re-scaled to a max RPS. Negative = astern.

**Sea currents.** **Yes**, via relative velocity. The polynomial inputs are
*ν_r*, not ground velocity. In practice this means swapping *(u, v)* for
*(u, v)* minus the body-frame projection of *V_c* before each polynomial
evaluation.

**Sources.**
- Abkowitz (1964), *Lectures on Ship Hydrodynamics — Steering and Manoeuvrability*.
- <https://www.researchgate.net/publication/241105238_Identification_of_Abkowitz_Model_for_Ship_Manoeuvring_Motion_Using_e_-Support_Vector_Regression>
- <https://www.tandfonline.com/doi/full/10.1080/17445302.2022.2067409>

---

### 1.4 Davidson–Schiff linear manoeuvring

The 1946 origin of every later 3-DoF model. **Linear-only approximation** of
sway and yaw dynamics around a straight-ahead trim, with surge speed held
constant.

**Symbols.**
- **x, y, ψ** — pose.
- **U₀** — assumed-constant forward speed.
- **v, r** — sway and yaw velocity (small perturbations).
- **δ** — rudder angle (small).

**Interaction.** At a fixed speed *U₀*, a small rudder deflection produces a
small side-force and a small yaw moment that linearly drive *v* and *r*. The
sway-yaw subsystem is two coupled first-order lags with linear coefficients
that come from captive-model tests. Good only near straight-line motion;
breaks down for sharp turns or speed changes.

**Action space.** One or two dimensions:
- **a₀ — rudder angle** *δ*: small-perturbation rudder command. Normalised to
  `[-1, 1]`, re-scaled to ±δ_max (kept small to stay in the linear regime).
- **a₁ — surge thrust** (optional): only used if `U₀` is allowed to drift.
  Normalised to `[-1, 1]`. Often omitted because the model assumes constant
  *U₀*.

**Sea currents.** Possible but rarely used. Currents would enter as a constant
additive offset to *(v)* (the body-frame sway disturbance) and to the world-
frame position integration. Because the model assumes small perturbations, a
large current changes the operating point and invalidates the linearisation.

**Sources.**
- Davidson & Schiff (1946), *Turning and course-keeping qualities*, Stevens Institute of Technology.
- <https://arxiv.org/html/2502.18696v1>

---

## 2. Ship-specific steering-response models (1-DoF yaw)

### 2.1 Nomoto 1st-order (KT model)

A drastic simplification: ignores everything except how the **yaw rate
responds to rudder** at a roughly constant forward speed. Two parameters
identify the whole ship.

**Symbols.**
- **x, y, ψ** — pose.
- **r** — yaw rate.
- **U** — forward speed (parameter, not a state).
- **δ** — rudder angle.
- **K** — gain (how strongly rudder turns the ship eventually).
- **T** — time constant (how slowly the ship responds to rudder).

**Interaction.** Turning the rudder is like commanding a target yaw rate
through a first-order lag of time constant *T*. The ship eventually rotates
at *K · δ*. The heading is the integral of *r*. World-frame motion is a
straight line at speed *U* along whatever direction *ψ* currently points. No
sway, no surge dynamics, no inertia coupling. Suitable for autopilot design
and as a tiny surrogate for very large fleets.

**Action space.** One dimension:
- **a₀ — rudder angle** *δ*: how far to turn the rudder. Normalised to
  `[-1, 1]`, re-scaled to ±δ_max. Negative = port, positive = starboard.
  Forward speed *U* is a parameter, not an action; surge cannot be controlled.

**Sea currents.** Treated as a passive add-on. Currents do not affect the
rudder-to-yaw response (the model has no sway state to disturb). They are
added directly to the world-frame *(ẋ, ẏ)* as drift. This is a kinematic
hack, not physically correct, but acceptable for traffic-scale simulation.

**Sources.**
- Nomoto, Taguchi, Honda, Hirano (1957), *On the steering qualities of ships*.
- <https://www.researchgate.net/publication/265324077_Fundamental_Properties_of_Linear_Ship_Steering_Dynamic_Models>
- <https://www.researchgate.net/publication/277497099_Ships_Steering_Autopilot_Design_by_Nomoto_Model>

---

### 2.2 Nomoto 2nd-order (Nomoto–Bech)

Same idea as Nomoto-1 but **adds the overshoot you see in real ships**: the
yaw rate does not settle exponentially, it slightly oscillates before
settling, because sway and yaw are coupled.

**Symbols.**
- **x, y, ψ** — pose.
- **r** — yaw rate.
- **ṙ** — yaw acceleration (a separate state for the 2nd-order dynamics).
- **U** — forward speed (parameter).
- **δ** — rudder angle.
- **T₁, T₂** — two time constants (slow and fast yaw response).
- **T₃** — rudder-derivative time constant (lead term).
- **K** — gain.

**Interaction.** Two cascaded first-order lags from rudder to yaw rate, with
a small rudder-derivative feed-forward. Captures the "ship leans into the
turn" overshoot. In practice the two time constants are often hard to
identify separately — they appear close to a zero in the response, and
small data noise makes the fit unstable.

**Action space.** One dimension:
- **a₀ — rudder angle** *δ*: same meaning as Nomoto-1, normalised to
  `[-1, 1]` and re-scaled to ±δ_max.

**Sea currents.** Same passive add-on as Nomoto-1: drift is added to world-
frame velocity; the rudder-yaw subsystem ignores the current.

**Sources.**
- <https://www.researchgate.net/publication/265324077_Fundamental_Properties_of_Linear_Ship_Steering_Dynamic_Models>
- <https://shipjournal.co/index.php/sst/article/view/160>

---

### 2.3 Norrbin nonlinear steering

Nomoto-1 plus a **polynomial in yaw rate** to capture course-unstable
behaviour seen in big tankers and bulk carriers.

**Symbols.**
- **x, y, ψ** — pose.
- **r** — yaw rate.
- **U** — forward speed (parameter).
- **δ** — rudder angle.
- **K** — gain.
- **T** — time constant.
- **n₀, n₁, n₂, n₃** — coefficients of the yaw-rate polynomial; the cubic term
  captures large-angle nonlinearity and course instability.

**Interaction.** Identical structure to Nomoto-1 except the linear damping is
replaced with a polynomial that allows the ship to "fall off course" when
disturbed — a real phenomenon for slow, full-form ships. Better fit for
tankers and bulk carriers; mild added cost.

**Action space.** One dimension:
- **a₀ — rudder angle** *δ*: same meaning as Nomoto-1, normalised to
  `[-1, 1]` and re-scaled to ±δ_max. The nonlinear damping lives in the
  dynamics, not in the action.

**Sea currents.** Same passive add-on as Nomoto-1: drift added to world-frame
velocity.

**Sources.**
- Norrbin (1963), *On the design and analysis of zig-zag tests*.
- <https://www.mdpi.com/2077-1312/13/3/534>
- <https://www.sciencedirect.com/science/article/abs/pii/S0029801821016796>

---

## 3. 6-DoF ship models (seakeeping)

### 3.1 Fossen 6-DoF

The full rigid-body picture. Adds **vertical motions and tilting** to the
horizontal Fossen 3-DoF model. Needed when waves, roll stability, or
seakeeping matter.

**Symbols.**
- **x, y, z** — world-frame position (now including depth).
- **φ (phi)** — roll: rotation about the ship's length axis (lean side-to-side).
- **θ (theta)** — pitch: rotation about the beam axis (bow up / down).
- **ψ** — yaw.
- **u, v, w** — body-frame linear velocities (surge, sway, heave).
- **p, q, r** — body-frame angular velocities (roll, pitch, yaw rates).
- **τ** — 6-D control wrench.
- **τ_wave** — wave excitation, treated as exogenous.
- **V_c, ν_r** — current vector and relative velocity through water.

**Interaction.** Same rigid-body equations as Fossen 3-DoF, lifted to six
dimensions. The hull has hydrostatic restoring forces in heave, roll and
pitch (gravity + buoyancy try to keep it upright and at waterline). Waves
add an exogenous wrench *τ_wave* on top. Coupling is dense: a yaw-moment
rudder force can cause a small roll, a wave hitting the bow excites pitch
and heave simultaneously, and roll changes the effective rudder-to-yaw gain.
Used for stabiliser-fin control, parametric roll, green-water studies.

**Action space.** Six dimensions (the components of *τ*):
- **a₀ — surge force** *τ_X*: forward / aft push along length axis.
- **a₁ — sway force** *τ_Y*: lateral push across beam.
- **a₂ — heave force** *τ_Z*: vertical push (ballast, hydrofoil — rare for
  surface ships, common for submarines).
- **a₃ — roll moment** *τ_K*: torque about the length axis (stabiliser fins,
  anti-roll tanks).
- **a₄ — pitch moment** *τ_M*: torque about the beam axis (trim tabs, ballast
  shift).
- **a₅ — yaw moment** *τ_N*: turning torque about the vertical axis.

Each normalised to `[-1, 1]` and re-scaled to per-axis max. Most surface
ships zero-out *a₂, a₃, a₄* — actuators do not exist for those axes.

**Sea currents.** Yes, same treatment as Fossen 3-DoF: subtract *V_c* from the
ground velocity before feeding hydrodynamic damping and Coriolis. Currents
typically have negligible direct effect on vertical motions (*z, φ, θ*) but
do affect surge / sway / yaw exactly as in 3-DoF.

**Sources.**
- <https://onlinelibrary.wiley.com/doi/book/10.1002/9781119994138>
- <https://www.researchgate.net/publication/362750653_Data-driven_simultaneous_identification_of_the_6DOF_dynamic_model_and_wave_load_for_a_ship_in_waves>
- <https://journals.sagepub.com/doi/full/10.1177/14750902231157852>
- <https://www.nature.com/articles/s41598-026-46427-8>

---

## 4. Underactuated USV

Same equations as Fossen 3-DoF, but the **agent cannot push sideways directly**
— there is no body-frame sway thruster. Only surge thrust and yaw moment are
available. Matches almost all real autonomous-surface-vessel hardware.

**Symbols.**
- **x, y, ψ** — pose.
- **u, v, r** — surge / sway / yaw velocities.
- **τ_X** — surge thrust (forward push).
- **τ_N** — yaw moment (turning torque).
- **V_c, ν_r** — current vector and relative velocity.

**Interaction.** The agent controls only forward thrust and steering moment.
Sway velocity *v* still arises, but only as a passive consequence of yaw
(the hull skids while turning) and of current disturbance. This is the same
"non-holonomic" constraint that makes cars harder to park than holonomic
robots. Control design becomes harder because sway cannot be regulated
directly — typical solutions use line-of-sight guidance and let sway decay
naturally through damping.

**Action space.** Two dimensions:
- **a₀ — surge thrust** *τ_X*: forward push from the propeller / waterjet.
  Normalised to `[-1, 1]`, re-scaled to a max thrust. Negative = astern.
- **a₁ — yaw moment** *τ_N*: steering torque, produced either by a rudder
  behind the propeller or by differential thrust between port and starboard
  thrusters. Normalised to `[-1, 1]`.

Alternative low-level parameterisation when the actuators are exposed
directly: `(δ, n)` for rudder-screw vessels, or `(n_port, n_stbd)` for
differential-thruster catamarans / ASVs.

**Sea currents.** Yes, identical to Fossen 3-DoF: currents enter through
*ν_r*. They are especially important here because the underactuated ship
cannot cancel sideways drift directly; it has to yaw into the current and
trade some forward speed to compensate.

**Sources.**
- <https://www.nature.com/articles/s41598-022-05456-9>
- <https://pmc.ncbi.nlm.nih.gov/articles/PMC6539673/>
- <https://www.sciencedirect.com/science/article/abs/pii/S0029801821013688>
- <https://journals.sagepub.com/doi/full/10.1177/1729881419871807>
- <https://onlinelibrary.wiley.com/doi/10.1002/cpe.6567>

---

## 5. Generic / kinematic fallbacks

### 5.1 Constant Velocity (CV)

Models the ship as **a point moving in a straight line at fixed velocity**.
Useful only for AIS replay and as a baseline for trajectory prediction.

**Symbols.**
- **x, y** — position.
- **vx, vy** — world-frame velocity components.

**Interaction.** No interaction. Position is the integral of velocity, and
velocity does not change unless the higher-level scenario script changes it.

**Action space.** Zero or two dimensions:
- *Passive mode:* no action — the trajectory is scripted.
- *Driven mode:* `(a₀, a₁) = (vx_cmd, vy_cmd)` — commanded world-frame
  velocity components, normalised to `[-1, 1]` and re-scaled to a max speed.

**Sea currents.** Trivial — currents simply add to the velocity each step.

**Sources.**
- <https://www.cambridge.org/core/journals/journal-of-navigation/article/abs/simplified-simulation-model-of-ship-navigation-for-safety-and-collision-avoidance-in-heavy-traffic-areas/D8671045BDAC618F5330FE4E44BB4848>

---

### 5.2 CTRV — Constant Turn-Rate and Velocity

Models the ship as **moving along a circular arc** at constant speed and
constant turn rate between steps.

**Symbols.**
- **x, y, ψ** — pose.
- **v** — scalar speed.
- **ω (omega)** — turn rate.

**Interaction.** The ship sweeps an arc whose radius is *v / ω*. Setting
*ω = 0* collapses to a straight line. Common in multi-object tracking filters
and for plausible AIS interpolation.

**Action space.** Zero or two dimensions:
- *Passive mode:* no action — the ship sweeps a fixed arc.
- *Driven mode:* `(a₀, a₁) = (v_cmd, ω_cmd)` — commanded scalar speed and
  turn rate, both normalised to `[-1, 1]` and re-scaled to per-axis maxima.

**Sea currents.** Treated as a passive add-on to the world-frame velocity,
just like Nomoto. The arc shape itself does not know about currents.

**Sources.**
- General reference: same as §5.1; widely used in tracking literature.

---

### 5.3 Unicycle / DiffDrive (already implemented)

Models the ship as a **two-wheel kinematic robot**: forward speed and turn
rate are commanded independently, no sideways slip.

**Symbols.**
- **x, y, ψ** — pose.
- **agent_vel** — world-frame velocity (a derived view).
- **u_lin** — commanded forward speed.
- **u_ang** — commanded turn rate.

**Interaction.** A toy approximation: the ship turns instantly without
inertia and slides forward at the commanded speed. Already in CAMAR as
`DiffDriveDynamic`.

**Action space.** Two dimensions:
- **a₀ — forward speed command** *u_lin*: commanded scalar speed along the
  body-frame forward axis. Normalised to `[-1, 1]`, re-scaled to a max speed.
- **a₁ — turn-rate command** *u_ang*: commanded yaw rate. Normalised to
  `[-1, 1]`, re-scaled to a max angular speed.

**Sea currents.** Can be added as an offset to the world-frame velocity. The
non-holonomic constraint (no body-frame sideways motion) is violated by
currents — but for kinematic fallbacks this is acceptable.

**Sources.** (No external citation — standard robotics textbook.)

---

## 6. Data-driven / hybrid models

### 6.1 Fossen base + neural / GP residual

Takes a Fossen 3-DoF or MMG base model and **adds a learned correction** to
absorb whatever the physics misses (un-modelled hull effects, hull fouling,
wind windage, mis-identified coefficients).

**Symbols.**
- **x, y, ψ, u, v, r** — physical state (same as the base model).
- **τ** — control wrench.
- **f_θ** — learned residual function with parameters *θ* (a neural net or
  Gaussian process).

**Interaction.** The base physics model produces a nominal next-state. The
residual model takes the current state and action and outputs a small
correction added to that next-state. With enough data on a specific hull,
this can match real-world manoeuvres closely. The base physics keeps the
model from extrapolating crazily in regions the data did not cover.

**Action space.** Same as the underlying base model — typically Fossen 3-DoF
*(τ_X, τ_Y, τ_N)* or MMG *(δ, n)*. The residual sees the action as one of
its inputs but does not change the action interface.

**Sea currents.** Inherited from the base model. The residual can also learn
to compensate for un-modelled current effects if current information is fed
in as an input, but typically the base model handles currents and the
residual handles everything else.

**Sources.**
- <https://html.rhhz.net/jmsa/html/20160407.htm>
- <https://www.mdpi.com/2073-8994/13/10/1956>
- <https://arxiv.org/html/2502.18696v1>
- <https://link.springer.com/article/10.1007/s00773-024-01045-9>

---

### 6.2 Pure neural / Koopman surrogate

A learned simulator: a **sequence model** (LSTM, Transformer, Koopman
operator) that maps current state and action to next state with no physics
prior.

**Symbols.**
- **z** — latent state vector learned by the model (typically 16–64
  dimensions).
- **(x, y, ψ)** — optionally decoded from *z* or kept alongside.

**Interaction.** Whatever the network has learned. Fast at inference, no
explicit hydrodynamic interpretation, hard to inverse-engineer. Useful as a
"digital twin" of a specific hull when training data is plentiful.

**Action space.** Whatever the training data uses. Most papers mirror the
ship-specific dynamic they replaced: two dimensions for `(δ, n)`, three for
`(τ_X, τ_Y, τ_N)`, or six for the full Fossen wrench. The action vector is
just another input to the network — its physical meaning is whatever the
labels were during training.

**Sea currents.** Implicit — the model only knows about currents if current
information was provided during training and at inference. Typical
implementations feed a current vector into the input alongside state and
action, and trust the network to learn the coupling.

**Sources.**
- <https://www.sciencedirect.com/science/article/abs/pii/S0029801823016396>
- <https://journals.sagepub.com/doi/full/10.1177/14750902231157852>
