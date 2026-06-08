# 2. Task-Space PD Control

This exercise extends feedback control from joint space to **task space** (operational space) for the same two-joint planar robotic arm simulated in MuJoCo. Instead of regulating joint angles, the controller now drives the **end-effector** toward a desired Cartesian position.

---

## What is different now?

This script implement a PD controller that drives the arm's end-effector (the `tip` body) from its current Cartesian position to a desired target position in the plane.

The controller computes a force in task space and maps it back to joint torques. Instead of rotational springs at the joints, this task conceptualizes the control as linear springs connecting the tip of the fingers to the target.

> [!TIP]
> You can move the task-space target by Ctrl+Right-click dragging it around in the interactive window.
> Alternatively you can add the `use_traj=True` arguement to the `get_current_and_target_kinematics` function's call on line 26 to automate the target movement.

---

## Control Law

The task-space PD controller is, familiarly, defined as:

$$
F = K_p \, e_{pos} + K_d \, e_{vel}
$$

where:

* $e_{pos} = x_{target} - x$ is the Cartesian position error
* $e_{vel} = -\dot{x}$ is the Cartesian velocity damping term
* $K_p$ is the proportional gain
* $K_d$ is the derivative gain
* $F$ is the desired force at the end-effector

The Cartesian force is converted into joint torques using the transpose of the Jacobian:

$$
\tau = J^\top F
$$

You might notice that in terms of forces applied this is very similar to the joint space control. However,
since the inertia of the arm still varies based on joint configuration, this control will still not behave linearly in task space.

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

## Key Insights

* This is a **task-space controller**: errors and forces are expressed in Cartesian coordinates, not joint angles
* The Jacobian is **configuration-dependent** and must be recomputed every frame
* $J^\top$ maps Cartesian forces to joint torques, avoiding explicit inverse kinematics
* Near **singular configurations** the Jacobian loses rank and the mapping degrades
* A zero derivative gain removes all task-space damping and tends to produce oscillation


## Source Code

[View `02_pd_task_space.py` on GitHub](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/02_pd_task_space.py)  
[View solution `02_pd_task_space.py` on GitHub](https://github.com/Balint-H/ssnr_sim/blob/main/SSNR2026/solutions/02_pd_task_space.py)

```{literalinclude} ../SSNR2026/02_pd_task_space.py
:language: python
:linenos:
:caption: Task-space PD control implementation in MuJoCo
```








