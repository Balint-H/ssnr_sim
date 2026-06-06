# 2. Task-Space PD Control

This exercise extends feedback control from joint space to **task space** (operational space) for the same two-joint planar robotic arm simulated in MuJoCo. Instead of regulating joint angles, the controller now drives the **end-effector** toward a desired Cartesian position.

---

## Objective

Your task is to implement a PD controller that drives the arm's end-effector (the `tip` body) from its current Cartesian position to a desired target position in the plane.

The controller computes a force in task space and maps it back to joint torques.

---

## Control Law

The task-space PD controller is defined as:

$$
F = K_p \, e_{pos} + K_d \, e_{vel}
$$

where:

* $e_{pos} = x_{target} - x$ is the Cartesian position error
* $e_{vel} = -\dot{x}$ is the Cartesian velocity damping term
* $K_p$ is the proportional gain
* $K_d$ is the derivative gain
* $F$ is the desired force at the end-effector

The Cartesian force is then converted into joint torques using the transpose of the Jacobian:

$$
\tau = J^\top F
$$

---

## The Jacobian

The Jacobian $J$ is the linear map relating joint-space differences to task-space differences:

$$
\dot{x} = J(q)\,\dot{q}
$$

It depends on the **current configuration** of the arm, so it must be recomputed at every timestep. MuJoCo provides it via the built-in function `mujoco.mj_jac`.

The Jacobian is used twice in this controller:

* **Forward mapping** ($\dot{x} = J\dot{q}$): to obtain the end-effector velocity from joint velocities, needed for the damping term.
* **Transpose mapping** ($\tau = J^\top F$): to convert the desired Cartesian force into joint torques.

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

| Variable       | Description                                          |
| -------------- | ---------------------------------------------------- |
| `qpos`         | Joint positions (angles in radians)                  |
| `qvel`         | Joint velocities                                     |
| `J`            | End-effector Jacobian, mapping joint to task space   |
| `tip`          | Body whose Cartesian position is being controlled    |
| `qfrc_applied` | Torques applied to the system                        |

---

## Implementation Notes

You only need to edit the function [`feedback_control`](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/student_functions.py#L6) in [`student_functions.py`](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/student_functions.py).

The controller receives:

* Cartesian position error
* Cartesian velocity error

and must return a **task-space force vector**. The conversion to joint torques via $J^\top$ is handled outside `feedback_control`, in the control callback.


---

## Key Insights

* This is a **task-space controller**: errors and forces are expressed in Cartesian coordinates, not joint angles
* The Jacobian is **configuration-dependent** and must be recomputed every frame
* $J^\top$ maps Cartesian forces to joint torques, avoiding explicit inverse kinematics
* Near **singular configurations** the Jacobian loses rank and the mapping degrades
* A zero derivative gain removes all task-space damping and tends to produce oscillation

---

```{note}
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

To run this exercise, in a new Colab notebook run:

    !git clone https://github.com/Balint-H/ssnr_sim.git
    %cd ssnr_sim
    !pip install mujoco
    !python -m SSNR2026.02_pd_task_space

The interactive viewer requires a local display and will not open on Colab.
```

## Source Code

[View `02_pd_task_space.py` on GitHub](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/02_pd_task_space.py)  
[View solution `02_pd_task_space.py` on GitHub](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/solutions/02_pd_task_space.py)

```{literalinclude} ../SSNR2026/02_pd_task_space.py
:language: python
:linenos:
:caption: Task-space PD control implementation in MuJoCo
```








