# SSNR 2026: WS4 – Dynamic Simulations of Neuromechanics

Welcome to the workshop materials for **WS4 – Dynamic Simulations of Neuromechanics**.

<div style="text-align:center; margin: 2rem 0;">
  <video width="700" controls>
    <source src="../_static/Media1.mp4" type="video/mp4">
  </video>
</div>

## Overview

A three-day workshop exploring how the nervous system controls movement,
from the biophysics of a single spiking neuron, through classical control
of a MuJoCo arm, to reinforcement learning agents that discover motor
strategies on their own.

```{admonition} What you will build
:class: tip

Day 1 — a LIF neuron pool that converts descending drive into muscle excitation.

Day 2 — a closed-loop controller where that excitation drives a MuJoCo arm
through PD control, tendons, Hill muscles, and motoneurons.

Day 3 — a PPO agent that learns to control the same arm from scratch,
using only a reward signal.
```


<div style="display:flex; justify-content:center; margin: 2rem 0;">
  <video 
    width="700" 
    autoplay 
    muted 
    loop 
    playsinline 
    controls
    preload="auto"
  >
    <source src="../_static/pipeline.mp4" type="video/mp4">
  </video>
</div>

### Day 1 — From Single LIF Neurons to Muscle Activation

A Jupyter notebook where you build a **Leaky Integrate-and-Fire (LIF)**
neuron from scratch, scale it to a pool, and convert discrete spikes
into smooth muscle excitation.

→ [Tutorial 1: The LIF Neuron](../SSNR2026/0_lif_neuron_exercises)

### Day 2 — From PD control to a neural-driven musculoskeletal model

A guided walkthrough of five MuJoCo scripts at increasing levels of
biological realism — joint torques → task space → tendons → muscles →
motoneuron pool.

→ [Tutorial 2: Controlling a Musculoskeletal Arm](day2_intro)

### Day 3 — Reinforcement Learning for Motor Control

Instead of designing a controller by hand, you specify only a **reward
signal** and let a PPO agent discover the behaviour through trial and
error — from a cartpole swing-up on CPU to a musculoskeletal elbow arm,
up to GPU-accelerated training with MJX and Brax.

→ [Tutorial 3: RL for Motor Control](day3_intro)

---

## Getting started

→ [Full setup instructions](getting_started)

---

## Materials

### Slides

| Day | Topic | Download |
|-----|-------|----------|
| Day 1 | Introduction & LIF Neuron | [DAY_1.pdf](../slides/DAY_1.pdf) |
| Day 2 | MuJoCo & Control | [DAY_2.pdf](../slides/DAY_2.pdf) |
| Day 3 | Reinforcement Learning | [RL.pdf](../slides/RL.pdf) |

### Further reading

- **MuJoCo** — tutorials, documentation, and community support:
  [github.com/google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)

- **De Groote, F. and Falisse, A., 2021.** Perspective on musculoskeletal modelling and predictive simulations of human movement to assess the neuromechanics of gait. *Proceedings of the Royal Society B*, 288(1946), p.20202432.

---

## Contact

For questions about the materials or technical issues, reach out to:

- **Claudia Sabatini** — [cn724@ic.ac.uk](mailto:cn724@ic.ac.uk)
- **Balint Hodossy** — [bkh16@ic.ac.uk](mailto:bkh16@ic.ac.uk)

Or open an issue on [GitHub](https://github.com/Balint-H/ssnr_sim/issues).
