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

You will need to edit the function [`feedback_control`](https://github.com/Balint-H/ssnr_sim/blob/main/src/ssnr_sim/SSNR2026/student_functions.py#L6) in [`student_functions.py`](https://github.com/Balint-H/ssnr_sim/blob/main/src/ssnr_sim/SSNR2026/student_functions.py).

The controller receives:

* position error
* velocity error

and must return a torque vector.

Unsure what gains you should be using? The value of the sliders in the interactive viewer is accessible in the `feedback_control` function. Use their values (e.g. from `data.control[2]`) as the gains for the PD control, and explore their effect interactively!

At a later point you will also need to edit the mujoco model files `arm_model.xml` and `arm_model_tendon.xml`.
You may do this with the text editor of your choice.

---

## Key Insights

* This is a **joint-space controller** (not task-space)
* Stability depends on the choice of gains $K_p$ and $K_d$
* Too high gains may cause oscillations
* The velocity term is essential for damping

---

```{note}
This exercise is designed to be ran locally, to be able to use the interactive viewer of MuJoCo.
Activate your virtual environment, then run each script separately e.g. `python 01_joint_space.py`  
```

## Source Code

[View `01_pd_joint_space.py` on GitHub](https://github.com/Balint-H/ssnr_sim/blob/main/src/ssnr_sim/SSNR2026/01_pd_joint_space.py)

```{literalinclude} ../src/ssnr_sim/SSNR2026/01_pd_joint_space.py
:language: python
:linenos:
:caption: PD control implementation in MuJoCo
```






