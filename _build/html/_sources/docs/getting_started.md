# Getting Started

The interactive simulations run in the open-source **MuJoCo** physics engine.
Follow the steps below to set up your environment before the workshop.

---

## Installation

### 1. Get the code

Clone the repository with Git (recommended — makes updating easy):

```bash
git clone https://github.com/Balint-H/ssnr_sim.git
cd ssnr_sim
```

No Git? Download the files by clicking the green **`<> Code ▾`** button on the GitHub page and extracting the archive.

### 2. Install uv

If you don't have a package manager yet, [install **uv**](https://docs.astral.sh/uv/getting-started/installation/) — it will collect all dependencies automatically.

Check it works:

```bash
uv --version
```

You should see something like `uv 0.x.x`.

### 3. Create the environment

From inside the repository folder, run:

```bash
uv sync
```

This creates and configures a virtual environment with all required packages.

### 4. Activate the environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

---

## Test your installation

Run the welcome scene — if you see a human shape relaxing in a hammock, you are ready:

```bash
python welcome_scene/hello_ssnr.py
```

```{admonition} macOS users
:class: note
macOS requires `mjpython` instead of `python` for any script that opens
the MuJoCo interactive viewer:
`mjpython welcome_scene/hello_ssnr.py`
```

---

## Running the workshop materials

**Day 1** — launch the notebook:

```bash
jupyter lab SSNR2026/0_lif_neuron_exercises.ipynb
```

**Day 2** — run each script from the terminal:

```bash
python SSNR2026/01_pd_joint_space.py
python SSNR2026/02_pd_task_space.py
python SSNR2026/03_pd_tendon_space.py
python SSNR2026/04_pd_muscle_space.py
python SSNR2026/05_pd_neuron_integration.py
```

**Day 3** — run the RL scripts from the `RL/` folder:

```bash
cd RL
python learn.py                  # train PPO on cartpole
python visualise_cartpole.py     # visualise the saved policy
python learn_myosuite.py         # train on the MyoSuite elbow arm
python visualise_elbow.py        # visualise the arm policy
```

The MJX notebook (`train_with_mjx.ipynb`) requires a CUDA-capable GPU.

---

## IDE setup (recommended)

Install an IDE such as **Spyder** or **PyCharm** to edit and debug scripts efficiently.
To install Spyder from the activated environment:

```bash
uv pip install spyder
```

```{admonition} IDE users
:class: note
If you use PyCharm or Spyder, read `info_for_ide_users.txt` in the repository root
to configure plotting correctly.
```

---

## Self-guided exploration

Open the `.xml` files in the `xml/` folder with your code editor — they are annotated
MuJoCo scenes with suggestions for things to try. You can edit `load_model.py` to
switch between scenes, or drag-and-drop an XML file onto an already-open MuJoCo viewer.

For a deeper dive, the official [MuJoCo tutorials](https://github.com/google-deepmind/mujoco)
are a great resource before or during the workshop.
