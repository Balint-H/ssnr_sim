# 3. Task-Space Control with Tendon Actuation

This exercise builds on task-space PD control and adds a layer of realism: the two-joint planar arm is no longer driven by torques applied directly at the joints, but by **tendons** (cables or muscles) that can only **pull**. The controller must therefore translate the desired Cartesian behavior all the way down to actuator (tendon) commands.

---

## Objective

Your task is to implement a PD controller that drives the arm's end-effector (the `tip` body) toward a desired Cartesian target, where the only available actuators are tendons.

The pipeline is: Cartesian force → joint torque → tendon force.

---

## Control Law

The task-space PD law is unchanged:

$$
F = K_p \, e_{pos} + K_d \, e_{vel}
$$

where:

* $e_{pos} = x_{target} - x$ is the Cartesian position error
* $e_{vel} = -\dot{x}$ is the Cartesian velocity damping term
* $F$ is the desired force at the end-effector

The Cartesian force is mapped to joint torques through the end-effector Jacobian transpose, and the joint torques are then mapped to tendon forces:

$$
\tau = J^\top F
\qquad\Longrightarrow\qquad
f_{tendon} = \left(J_{tendon}^\top\right)^{+} \tau
$$

---

## Two Jacobians

This controller chains **two** different Jacobians.

**End-effector Jacobian** $J$ relates joint velocities to task-space velocities:

$$
\dot{x} = J(q)\,\dot{q}
$$

It is used forward (for the task velocity) and as a transpose (to turn Cartesian force into joint torque), exactly as in the previous exercise.

**Tendon Jacobian** $J_{tendon}$ relates joint velocities to tendon length changes:

$$
\dot{L} = J_{tendon}(q)\,\dot{q}
$$

By the same force/velocity duality, tendon forces produce joint torques via its transpose, $\tau = J_{tendon}^\top f_{tendon}$. Inverting this relation to recover the tendon forces that realize a desired torque requires the **pseudoinverse** $\left(J_{tendon}^\top\right)^{+}$, since the system is typically over-actuated (more tendons than joints). The pseudoinverse selects the minimum-norm solution.

In MuJoCo the tendon Jacobian is stored in sparse form in `data.ten_J`; it is converted to a dense matrix with `mujoco.mju_sparse2dense` before use.

---

## The Pull-Only Constraint

Real tendons, cables, and muscles can only **pull**, never push. This means valid tendon forces are sign-constrained.

The unconstrained pseudoinverse solution may request negative (pushing) forces, which are physically impossible for a cable. The code includes a commented-out line showing how to clamp the result:

```python
# Muscles/cables should only pull, experiment with disabling pushing:
tendon_force = np.minimum(tendon_force, 0)
```

Enforcing this constraint changes the behavior: with antagonistic tendons, the controller must coordinate which cables pull to produce a net torque in the desired direction.

---

## Simulation Model

The robot is modeled as a rigid-body dynamical system governed by:

$$
M(q)\ddot{q} + C(q,\dot{q}) + g(q) = \tau
$$

MuJoCo solves these equations numerically at each timestep, with tendon forces entering through the actuation terms.

---

## Simulation Loop

At each timestep, MuJoCo performs the following steps:

1. Compute forward kinematics (body positions from joint states)
2. Call the control function (this is where your code runs)
3. Compute forward dynamics using the tendon-generated forces
4. Integrate system state forward in time

The control callback is executed every physics step (~2 ms).

---

## System Variables

| Variable        | Description                                              |
| --------------- | ------------------------------------------------------- |
| `qpos`          | Joint positions (angles in radians)                     |
| `qvel`          | Joint velocities                                        |
| `J`             | End-effector Jacobian, mapping joint to task space      |
| `ten_J`         | Tendon Jacobian (sparse), mapping joint to tendon space |
| `tip`           | Body whose Cartesian position is being controlled       |
| `ctrl`          | Tendon actuator commands (the forces you apply)         |
| `tendon_rgba`   | Per-tendon color, used here to visualize applied force  |

---

## Implementation Notes

You only need to edit the function [`feedback_control`](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/student_functions.py#L6) in [`student_functions.py`](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/student_functions.py).

The controller receives:

* Cartesian position error
* Cartesian velocity error

and must return a **task-space force vector**. Everything after that, the mapping to joint torques via $J^\top$, the conversion to tendon forces via the tendon Jacobian pseudoinverse, and writing to `data.ctrl`, is handled in the control callback.

The final block colors each tendon from red (pulling little / negative command) to blue (pulling hard) using a logistic mapping of `data.ctrl`, purely for visualization.

---

## Key Insights

* This is a **task-space controller** realized through **tendon (muscle) actuation**
* Two configuration-dependent Jacobians are chained: end-effector and tendon
* The tendon Jacobian is **sparse** in MuJoCo and must be densified before use
* A **pseudoinverse** is needed because the tendon system is usually over-actuated
* Tendons can only **pull**: enforcing this is what makes the model physically realistic
* Near kinematic singularities either Jacobian can become ill-conditioned

---

```{note}
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

To run this exercise, in a new Colab notebook run:

    !git clone https://github.com/Balint-H/ssnr_sim.git
    %cd ssnr_sim
    !pip install mujoco
    !python -m SSNR2026.03_pd_tendon_space

The interactive viewer requires a local display and will not open on Colab.
```

## Source Code

[View `03_pd_tendon_space.py` on GitHub](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/03_pd_tendon_space.py)  
[View solution `04_pd_tendon_space.py` on GitHub](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/solutions/04_pd_tendon_space.py)  
[View extended solution '05_pd_tendon_space_task_impedance.py' on GitHub](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/solutions/05_pd_tendon_space_task_impedance.py)  
[View extended solution '05_pd_tendon_space_task_stiffness.py' on GitHub](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/solutions/05_pd_tendon_space_task_stiffness.py)

```{literalinclude} ../SSNR2026/03_pd_tendon_space.py
:language: python
:linenos:
:caption: Tendon-actuated task-space control implementation in MuJoCo
```


