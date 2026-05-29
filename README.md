# SSNR Simulation Workshop

Clone the repository (if you have Git) or download the files directly by pressing the green button labelled `<> Code ▾`.
We strongly recommend using Git, since it will allow smoothly updating the project with the most recent changes.
You can install it [here](https://git-scm.com/install/) if you don't have it yet.

Follow the instructions below to set up a python environment ready to run the files and challenges.
If you use an IDE like PyCharm or Spyder, please also read the instructions in "info_for_ide_users.txt" to set up plotting correctly.


# MuJoCo Simulation Setup Guide

The interactive simulations will be executed in the open-source MuJoCo physics engine. To define the behaviour of the simulated systems, we'll use the Python API of MuJoCo. Please follow these steps to ensure you are able to run and edit scripts during the summer school.

## Setting up MuJoCo for forward dynamics

If you are already familiar with package/environment managers, you can use the manager of your choice.

If you don't have a package manager yet, [we recommend getting **uv**](https://docs.astral.sh/uv/getting-started/installation/). A package manager will help us collect all the code we need (e.g. the physics engine) in one place.

Test your installation by opening up a terminal / command window and typing the command:
```bash
uv --version
```

You should see a response like uv 0.x.x (or your current version of uv).

Installation Steps
Download and extract the files from the following repository: Balint-H/ssnr_sim (github.com)
You can use Git to clone it, or click on the <> Code ▾ button on the GitHub page to download it.

Navigate to the extracted folder with a terminal.

Sync the environment. Since the repository has the project file already set up, you can automatically create and configure your virtual environment by running:

``` bash
uv sync
```
Activate the created environment:

macOS / Linux:
``` bash
source .venv/bin/activate
```
Windows:
``` DOS
.venv\Scripts\activate
```
From now on, any installs made with this terminal will only affect this environment.

Testing Your Installation
We'll run a Python script that just opens a scene in MuJoCo's interactive viewer. If you see a human shape relaxing in a hammock, then you are ready for the workshop!

``` Bash
python welcome_scene/hello_ssnr.py
```

> [!NOTE]  
> Note for macOS Users: Due to how macOS handles graphics threads, you must use `mjpython instead of standard python to run scripts that open the interactive viewer. Run the test scene using:
> ```Bash
> mjpython welcome_scene/hello_ssnr.py
> ```

Development & Next Steps
(Recommended) IDE Setup
Install an Integrated Development Environment to efficiently edit and debug your scripts. We recommend getting Spyder or PyCharm. To install Spyder, you just need to run the following command from your activated environment:

``` bash
uv pip install spyder
```

(Suggested self guided study)
Open the .xml files included in the xml folder with a code editor. You can use your IDE to do so. Inside you'll find an annotated MuJoCo scene, breaking down individual elements simulated, and making suggestions of things to consider to try out and explore.

You can edit the load_model.py file to run the other scenes, or drag-and-drop the xml on to an already open MuJoCo viewer. If interested, looking at the official introductory MuJoCo tutorials is a good way to dig deeper before the workshop.

The 2026 simulation workshop for SSNR was organized by Claudia Sabatini and Balint Hodossy.

If you have further questions about the workshop and its contents, feel free to contact us at:
bkh16@ic.ac.uk
