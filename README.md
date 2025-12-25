![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/pre-commit.yaml?logo=pre-commit&logoColor=white&label=pre-commit&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/pre-commit.yaml)
[![docs status](https://img.shields.io/github/actions/workflow/status/isaac-sim/IsaacLab/docs.yaml?label=docs&color=brightgreen)](https://github.com/isaac-sim/IsaacLab/actions/workflows/docs.yaml)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)


**Isaac Lab** is a GPU-accelerated, open-source framework designed to unify and simplify robotics research workflows,
such as reinforcement learning, imitation learning, and motion planning. Built on [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html),
it combines fast and accurate physics and sensor simulation, making it an ideal choice for sim-to-real
transfer in robotics.

Isaac Lab provides developers with a range of essential features for accurate sensor simulation, such as RTX-based
cameras, LIDAR, or contact sensors. The framework's GPU acceleration enables users to run complex simulations and
computations faster, which is key for iterative processes like reinforcement learning and data-intensive tasks.
Moreover, Isaac Lab can run locally or be distributed across the cloud, offering flexibility for large-scale deployments.

A detailed description of Isaac Lab can be found in our [arXiv paper](https://arxiv.org/abs/2511.04831).

## Key Features

Isaac Lab offers a comprehensive set of tools and environments designed to facilitate robot learning:

- **Robots**: A diverse collection of robots, from manipulators, quadrupeds, to humanoids, with more than 16 commonly available models.
- **Environments**: Ready-to-train implementations of more than 30 environments, which can be trained with popular reinforcement learning frameworks such as RSL RL, SKRL, RL Games, or Stable Baselines. We also support multi-agent reinforcement learning.
- **Physics**: Rigid bodies, articulated systems, deformable objects
- **Sensors**: RGB/depth/segmentation cameras, camera annotations, IMU, contact sensors, ray casters.


## Getting Started

### Documentation

Our [documentation page](https://isaac-sim.github.io/IsaacLab) provides everything you need to get started, including
detailed tutorials and step-by-step guides. Follow these links to learn more about:

- [Installation steps](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)
- [Reinforcement learning](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_existing_scripts.html)
- [Tutorials](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/index.html)
- [Available environments](https://isaac-sim.github.io/IsaacLab/main/source/overview/environments.html)


## Isaac Sim Version Dependency

Isaac Lab is built on top of Isaac Sim and requires specific versions of Isaac Sim that are compatible with each
release of Isaac Lab. Below, we outline the recent Isaac Lab releases and GitHub branches and their corresponding
dependency versions for Isaac Sim.

| Isaac Lab Version             | Isaac Sim Version         |
| ----------------------------- | ------------------------- |
| `main` branch                 | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.3.X`                      | Isaac Sim 4.5 / 5.0 / 5.1 |
| `v2.2.X`                      | Isaac Sim 4.5 / 5.0       |
| `v2.1.X`                      | Isaac Sim 4.5             |
| `v2.0.X`                      | Isaac Sim 4.5             |


## BIMANUAL-MANIPULATION PIPELINE

# G1 Steering-Wheel Locomanipulation (IsaacLab)  
**End-to-end Mimic + Robomimic pipeline (datasets → BC → locomanip SDG)**

This repo/workspace guide shows the exact sequence of commands to:
1) collect (optional) teleop demos,  
2) annotate + expand them into a large **manipulation-only** dataset using **IsaacLab Mimic**,  
3) train a **behavior cloning (BC)** policy using **Robomimic**, and  
4) generate a **locomanipulation (navigation + manipulation)** dataset for the **G1 steering-wheel locomanipulation** environment (`Isaac-G1-SteeringWheel-Locomanipulation`).

---

## Table of Contents
- [What you get](#what-you-get)
- [Prerequisites](#prerequisites)
- [Project layout](#project-layout)
- [Quickstart](#quickstart)
- [Step-by-step](#step-by-step)
  - [1) Record manipulation demos (optional)](#1-record-manipulation-demos-optional)
  - [2) Annotate demos for Mimic](#2-annotate-demos-for-mimic)
  - [3) Generate a large manipulation dataset (Mimic SDG)](#3-generate-a-large-manipulation-dataset-mimic-sdg)
  - [4) Train a manipulation-only BC policy (Robomimic)](#4-train-a-manipulation-only-bc-policy-robomimic)
  - [5) Generate locomanipulation SDG dataset (walk + manipulate)](#5-generate-locomanipulation-sdg-dataset-walk--manipulate)
  - [6) Inspect navigation trajectories](#6-inspect-navigation-trajectories)
- [Common issues](#common-issues)
- [Outputs](#outputs)

---

## What you get

After completing the pipeline you will have:

- `generated_dataset_g1_locomanip.hdf5`  
  ✅ Large manipulation-only dataset (standing at table)

- Robomimic checkpoints (`*.pth`)  
  ✅ Manipulation-only BC policy you can roll out in the manipulation task

- `generated_dataset_g1_locomanipulation_sdg.hdf5`  
  ✅ Locomanipulation dataset (navigation + manipulation) built from the manipulation dataset

---

## Prerequisites

- IsaacLab installed and runnable via `./isaaclab.sh`
- Linux dependencies:
  ```bash
  sudo apt update
  sudo apt install -y cmake build-essential
  ```



* Optional but recommended:

  * `--enable_pinocchio` for better kinematics/IK stack
  * GPU supported; CPU also works (slower)

> **Important:** All commands below assume you run from the IsaacLab root folder.

---

## Project layout

Create a datasets directory in your IsaacLab root:

```text
IsaacLab/
├─ isaaclab.sh
├─ scripts/
│  ├─ tools/
│  └─ imitation_learning/
└─ datasets/
   ├─ dataset_g1_locomanip.hdf5
   ├─ dataset_annotated_g1_locomanip.hdf5
   ├─ generated_dataset_g1_locomanip.hdf5
   └─ generated_dataset_g1_locomanipulation_sdg.hdf5
```

---

## Quickstart

If you already have `dataset_annotated_g1_locomanip.hdf5` in `./datasets/`:

```bash
cd /home/jacob/Desktop/Projects/IsaacLab
mkdir -p datasets

# (A) generate large manipulation dataset
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu \
  --headless \
  --num_envs 20 \
  --generation_num_trials 1000 \
  --enable_pinocchio \
  --input_file ./datasets/dataset_annotated_g1_locomanip.hdf5 \
  --output_file ./datasets/generated_dataset_g1_locomanip.hdf5

# (B) train BC policy
./isaaclab.sh -i robomimic
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
  --algo bc \
  --normalize_training_actions \
  --dataset ./datasets/generated_dataset_g1_locomanip.hdf5

# (C) generate locomanipulation dataset (nav + manipulation)
./isaaclab.sh -p scripts/imitation_learning/locomanipulation_sdg/generate_data.py \
  --device cpu \
  --kit_args="--enable isaacsim.replicator.mobility_gen" \
  --task="Isaac-G1-SteeringWheel-Locomanipulation" \
  --dataset ./datasets/generated_dataset_g1_locomanip.hdf5 \
  --num_runs 1 \
  --lift_step 60 \
  --navigate_step 130 \
  --enable_pinocchio \
  --output_file ./datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
  --enable_cameras
```

---

## Step-by-step

### 0) Setup

```bash
cd /home/jacob/Desktop/Projects/IsaacLab   # <-- change to your IsaacLab root
mkdir -p datasets
```

---

### 1) Record manipulation demos (optional)

Record a few XR hand-tracking demos with G1 at the table:

```bash
./isaaclab.sh -p scripts/tools/record_demos.py \
  --device cpu \
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
  --teleop_device handtracking \
  --dataset_file ./datasets/dataset_g1_locomanip.hdf5 \
  --num_demos 5 \
  --enable_pinocchio
```

Replay to sanity-check:

```bash
./isaaclab.sh -p scripts/tools/replay_demos.py \
  --device cpu \
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
  --dataset_file ./datasets/dataset_g1_locomanip.hdf5 \
  --enable_pinocchio
```

If you **don’t** want to record demos, skip to Step 3 using an existing annotated dataset.

---

### 2) Annotate demos for Mimic

If you recorded your own dataset in Step 1, annotate it:

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu \
  --task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
  --input_file ./datasets/dataset_g1_locomanip.hdf5 \
  --output_file ./datasets/dataset_annotated_g1_locomanip.hdf5 \
  --enable_pinocchio
```

---

### 3) Generate a large manipulation dataset (Mimic SDG)

Generate many successful manipulation demos (standing at table):

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device cpu \
  --headless \
  --num_envs 20 \
  --generation_num_trials 1000 \
  --enable_pinocchio \
  --input_file ./datasets/dataset_annotated_g1_locomanip.hdf5 \
  --output_file ./datasets/generated_dataset_g1_locomanip.hdf5
```

**Key knobs**

* `--num_envs`: parallel generation environments
* `--generation_num_trials`: number of successful trials to generate
* `--headless`: faster, no GUI

---

### 4) Train a manipulation-only BC policy (Robomimic)

Install Robomimic:

```bash
./isaaclab.sh -i robomimic
```

Train BC on the generated manipulation dataset:

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
  --algo bc \
  --normalize_training_actions \
  --dataset ./datasets/generated_dataset_g1_locomanip.hdf5
```

#### Evaluate / play a checkpoint

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --device cpu \
  --enable_pinocchio \
  --task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
  --num_rollouts 50 \
  --horizon 400 \
  --norm_factor_min <NORM_FACTOR_MIN> \
  --norm_factor_max <NORM_FACTOR_MAX> \
  --checkpoint /PATH/TO/desired_model_checkpoint.pth
```

> `NORM_FACTOR_*` come from training output (normalization params).

---

### 5) Visualize the M3PO model (walk + manipulate)

Now generate the navigation+manipulation dataset for your env:

* **Task**: `Isaac-G1-SteeringWheel-Locomanipulation`
* **Input dataset**: `generated_dataset_g1_locomanip.hdf5`

```bash
./isaaclab.sh -p scripts/reinforcement_learning/m3po/play.py \
  --device cpu \
  --kit_args="--enable isaacsim.replicator.mobility_gen" \
  --task="Isaac-G1-SteeringWheel-Locomanipulation" \
  --dataset ./datasets/generated_dataset_g1_locomanip.hdf5 \
  --num_runs 1 \
  --lift_step 60 \
  --navigate_step 130 \
  --enable_pinocchio \
  --output_file ./datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
  --enable_cameras
```

**How to pick `lift_step` and `navigate_step`**

* `lift_step`: frame just after grasp + successful lift (object stable)
* `navigate_step`: frame where the robot is ready to start walking while holding the object

If generation fails or looks odd, these two values are the first things to tune.

---

### 6) Inspect navigation trajectories

Plot trajectories for quick debugging:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/m3po/plot_navigation_trajectory.py \
  --input_file ./datasets/generated_dataset_g1_locomanipulation_sdg.hdf5 \
  --output_dir /tmp/g1_nav_plots
```

---

## Common issues

### `./isaaclab.sh: No such file or directory`

You are not in the IsaacLab root. `cd` to the directory containing `isaaclab.sh`.

### `--kit_args="--enable isaacsim.replicator.mobility_gen"` ignored / nav gen fails

Ensure you used the exact `--kit_args` string. The locomanip SDG script depends on mobility gen being enabled.

### Low SDG success rate

* Bad demo quality (unstable grasps, inconsistent timing)
* `lift_step` / `navigate_step` misaligned
* Try generating fewer trials first (e.g., 50–100) to iterate quickly

### Pinocchio crashes

* Temporarily remove `--enable_pinocchio` to isolate the issue
* Confirm your IsaacLab environment has pinocchio installed/working

---

## Outputs

Expected outputs in `./datasets/`:

* `dataset_g1_locomanip.hdf5` (optional; recorded demos)
* `dataset_annotated_g1_locomanip.hdf5` (annotated demos)
* `generated_dataset_g1_locomanip.hdf5` (large manipulation dataset)
* `generated_dataset_g1_locomanipulation_sdg.hdf5` (locomanip dataset)

Robomimic training outputs:

* a run directory with logs and model checkpoints (`*.pth`)
* normalization parameters used by `play.py`



