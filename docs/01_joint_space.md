# 1. Joint-Space PD Control 

This exercise introduces a **Proportional-Derivative (PD) controller** for a two-joint planar robotic arm simulated in MuJoCo. It is the simplest form of feedback control and provides a foundation for understanding closed-loop dynamical systems.

---

## Objective

Your task is to implement a PD controller that drives the robotic arm from its current joint configuration to a desired target configuration.

The controller must compute the torque vector applied at each joint.

<div style="text-align:center; margin: 2rem 0;">
<img src="../_static/pd_loop.svg" width="840">
</div>

---

## Control Law

The PD controller is defined as:

$$
\tau = K_p , e_{pos} + K_d , e_{vel}
$$

where:

* $e_{pos} = q_{target} - q$ is the position error
* $e_{vel} = -\dot{q}$ is the velocity damping term
* $K_p$ is the proportional gain
* $K_d$ is the derivative gain
* $\tau$ is the applied joint torque

---

## Physical Interpretation

This controller is equivalent to a **spring-damper system in joint space**:

* The proportional term acts like a spring pulling the system toward the target
* The derivative term acts like a damper removing kinetic energy

Together, they stabilize the system and regulate motion.

---

## Simulation Model

The robot is modeled as a rigid-body dynamical system governed by:

$$
M(q)\ddot{q} + C(q,\dot{q}) + g(q) = \tau
$$

MuJoCo solves these equations numerically at each timestep.

---

## Simulation Loop

At each timestep, MuJoCo performs the following steps:

1. Compute forward kinematics (body positions from joint states)
2. Call the control function (this is where your code runs)
3. Compute forward dynamics using applied torques
4. Integrate system state forward in time

The control callback is executed every physics step (~2 ms).

---

## System Variables

| Variable       | Description                         |
| -------------- | ----------------------------------- |
| `qpos`         | Joint positions (angles in radians) |
| `qvel`         | Joint velocities                    |
| `ctrl`         | Desired target joint configuration  |
| `qfrc_applied` | Torques applied to the system       |

---

## Implementation Notes

You only need to edit the function [`feedback_control`](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/student_functions.py#L6) in [`student_functions.py`](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/student_functions.py).

The controller receives:

* position error
* velocity error

and must return a torque vector.

---

## Key Insights

* This is a **joint-space controller** (not task-space)
* Stability depends on the choice of gains $K_p$ and $K_d$
* Too high gains may cause oscillations
* The velocity term is essential for damping

---

## Source Code

[View `01_pd_joint_space.py` on GitHub →](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/01_pd_joint_space.py)

```{literalinclude} ../SSNR2026/01_pd_joint_space.py
:language: python
:linenos:
:caption: PD control implementation in MuJoCo
```

---

## Warning

```{warning}
High proportional gains may lead to unstable or oscillatory behavior.
Insufficient derivative damping may cause overshooting and persistent motion.
```


