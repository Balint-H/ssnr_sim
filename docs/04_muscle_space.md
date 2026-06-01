# 4. Operational-Space Control with Muscle Actuation

This final exercise assembles the full pipeline: task-space PD control, mapped to joint torques, then to **muscle** (Hill-type tendon) activations on a planar arm. It also adds the practical machinery needed to make this work robustly: target reachability clipping, force limiting, and an inertia-weighted Jacobian inverse. Unlike the earlier exercises, the muscle actuators are built **programmatically** by editing the model specification at load time.

---

## Objective

Drive the arm's end-effector (the `tip` body) toward an **interactive mocap target** the user can drag in the plane, using only muscles that pull. The controller implements the complete chain from Cartesian error to muscle activation.

---

## Control Law

The task-space PD law is applied inline (this exercise contains the full reference implementation rather than a stub to fill in):

$$
F = K_p \, e_{pos} - K_d \, \dot{x}
$$

with $K_p = 10$ and $K_d = 0.3$ — note that, unlike the previous exercises, this one **does include damping**. The Cartesian force is mapped to joint torques and then clipped:

$$
\tau = \mathrm{clip}\!\left(J^\top F,\; -1,\; 1\right)
$$

and finally resolved into muscle forces through the tendon Jacobian pseudoinverse:

$$
f_{tendon} = \left(J_{tendon}^\top\right)^{+}\tau,
\qquad
\text{ctrl} = -\min(0,\, f_{tendon})
$$

The last step keeps only the **pulling** component of each tendon and writes it as a non-negative activation command.

---

## The Interactive Mocap Target

The target is a **mocap body**, read from `data.mocap_pos[0]`. Because of this, the gesture **Ctrl (Cmd on Mac) + Shift + Right-click drag** genuinely moves the target in the horizontal plane — the controller reads the dragged position every frame. The z-coordinate is forced to zero to keep motion planar.

---

## Target Reachability Clipping

If the requested target lies outside the arm's reach, the inverse mappings produce erratic behavior. The code computes the total reach from the link offsets (`forearm`, `wrist_body`, `hand`, `tip`) and clips the target radius to that maximum while preserving direction:

$$
[x_t, y_t] \leftarrow \frac{[x_t, y_t]}{r_t}\,\mathrm{clip}\!\left(r_t,\, 0,\, \ell_s + \ell_e + \ell_h + \ell_w\right)
$$

This keeps the commanded target on or inside the reachable workspace.

---

## Inertia-Weighted Jacobian Inverse

The mass (inertia) matrix $H = M(q)$ is obtained with `mujoco.mj_fullM`, lightly regularized on the diagonal (`+0.01`) for numerical conditioning, and normalized by the upper-arm subtree mass.

It feeds a **weighted pseudoinverse**:

$$
J^{+}_H = H^{-1} J^\top \left(J H^{-1} J^\top\right)^{+}
$$

implemented in [`weighted_pinv`](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/04_pd_muscle_space.py#L17). This is the operational-space inverse that respects the arm's inertia, mapping task-space quantities into joint space in a dynamically consistent way.


---

## Muscle Actuators (programmatic model editing)

Rather than editing the XML by hand, the model is modified in [`load_callback`](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/04_pd_muscle_space.py#L87) using `mujoco.MjSpec`. Each ideal tendon actuator is converted into a **muscle** with `a.set_to_muscle(...)`, configuring the force-length and force-velocity curves (`lmin`, `lmax`, `vmax`, `fpmax`, `fvmax`), activation dynamics (`timeconst`), scaling, and a `ctrlrange` of `[0, 1]`.

This means the actuator command is a normalized **muscle activation**, not a raw force — the muscle model converts activation, length, and velocity into the actual contractile force.

---

## The Pull-Only Constraint

Muscles can only pull. The line

```python
data.ctrl = -np.minimum(0, tendon_force)
```

keeps only the negative (pulling) part of each requested tendon force and flips its sign to a non-negative activation. Pushing components are discarded, so producing a net torque in a given direction relies on the appropriate **antagonist** muscles activating.

---

## Simulation Model

The robot is a rigid-body dynamical system governed by:

$$
M(q)\ddot{q} + C(q,\dot{q}) + g(q) = \tau
$$

with the torque $\tau$ supplied by the muscle forces. MuJoCo solves these equations numerically each timestep.

---

## Simulation Loop

At each timestep, MuJoCo performs the following steps:

1. Compute forward kinematics (body positions from joint states)
2. Call the control function (this is where your code runs)
3. Compute forward dynamics, including muscle activation dynamics and forces
4. Integrate system state forward in time

The control callback is executed every physics step (~2 ms).

---

## System Variables

| Variable          | Description                                              |
| ----------------- | ------------------------------------------------------- |
| `qpos` / `qvel`   | Joint positions and velocities                          |
| `mocap_pos`       | Interactive target position (dragged by the user)       |
| `J`               | End-effector Jacobian, joint → task space               |
| `qM` / `mj_fullM` | Mass (inertia) matrix in sparse / dense form            |
| `ten_J`           | Tendon Jacobian (sparse), joint → tendon space          |
| `ctrl`            | Muscle activation commands, range `[0, 1]`              |
| `tendon_rgba`     | Per-tendon color, used to visualize applied force       |

---


## Key Insights

* This is **operational-space control** delivered through a realistic **muscle** model
* Two configuration-dependent Jacobians are chained (end-effector and tendon)
* The **inertia-weighted** pseudoinverse gives dynamically consistent task-to-joint mapping (available but inactive by default here)
* **Reachability clipping** prevents instability when the target is outside the workspace
* **Force clipping** keeps joint torques within reasonable bounds
* Muscles only **pull**, and the command is an **activation** in `[0, 1]`, not a raw force
* Models can be edited **programmatically** with `MjSpec`, avoiding manual XML changes

---

## Source Code

[View `04_pd_muscle_space.py` on GitHub →](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/04_pd_muscle_space.py)

```{literalinclude} ../SSNR2026/04_pd_muscle_space.py
:language: python
:linenos:
:caption: Muscle-actuated operational-space control in MuJoCo
```

---

## Warning

```{warning}
High proportional gains may lead to unstable or oscillatory behavior.
Insufficient derivative damping may cause overshooting and persistent motion.
Targets outside the reachable workspace cause erratic behavior unless clipped.
Near kinematic singularities the Jacobians and the (weighted) pseudoinverse become
ill-conditioned and can request very large forces.
Because muscles can only pull, some task-space directions may be under-actuated in
certain configurations.
```



