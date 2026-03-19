# Coding Agents Guide

This document describes how autonomous coding agents should work in this repository:
how to install and run code, how to treat "tests", and the expected Python code style.

The project simulates a 2-DOF robotic arm in PyBullet, with a custom Gym environment
and PPO training/evaluation scripts.

---

## 1. Environment and Dependencies

- Python: target Python 3.8+.
- OS: developed on desktop (Linux/Windows ok); PyBullet requires a graphical backend
  for GUI runs.
- Core libraries:
  - gym (legacy API, for example 0.21; env.step() returns (obs, reward, done, info)).
  - pybullet, pybullet_data.
  - numpy.
  - stable_baselines3 (PPO).
- Installation (main path):
  - Create and activate a virtualenv.
  - Install:
    ```bash
    pip install -r requirements.txt
    ```
- There is also requirements_legacy.txt for older environments; do not change or
  delete it unless explicitly requested.

---

## 2. Build / Run / Test Commands

There is no traditional build step; this is a pure Python project.

### 2.1. Running the main simulation

- Start the basic 2-link arm simulation (PyBullet GUI):
  ```bash
  python main.py
  ```

### 2.2. Training RL agents

- Train PPO with the custom environment in "pure" reward mode:
  ```bash
  python train_rl.py --mode pure --seed 0 --steps 300000
  ```
- Train PPO with the PIRL-style reward:
  ```bash
  python train_rl.py --mode pirl --seed 0 --steps 300000
  ```
- Recommended sweep (examples from README.md):
  ```bash
  # pure
  python train_rl.py --mode pure --seed 0 --steps 300000
  python train_rl.py --mode pure --seed 1 --steps 300000
  python train_rl.py --mode pure --seed 2 --steps 300000
  python train_rl.py --mode pure --seed 3 --steps 300000
  python train_rl.py --mode pure --seed 4 --steps 300000

  # pirl
  python train_rl.py --mode pirl --seed 0 --steps 300000
  python train_rl.py --mode pirl --seed 1 --steps 300000
  python train_rl.py --mode pirl --seed 2 --steps 300000
  python train_rl.py --mode pirl --seed 3 --steps 300000
  python train_rl.py --mode pirl --seed 4 --steps 300000
  ```
- Outputs:
  - runs_pure/seed_<seed>/monitor.csv and ppo_model_<seed>.zip.
  - runs_pirl/seed_<seed>/monitor.csv and ppo_model_<seed>.zip.
- Do not change these directory patterns unless you also update all downstream
  code and documentation.

### 2.3. Evaluating trained models

- Evaluate a single mode:
  ```bash
  python eval_rl.py --mode pure --episodes 100 --seed 0
  python eval_rl.py --mode pirl --episodes 100 --seed 0
  ```
- Evaluate both modes and compare (recommended):
  ```bash
  python eval_rl.py --mode both --episodes 100 --out results/eval_results --seed 0
  ```
- Example with rendering and per-episode logging (slower, for debugging):
  ```bash
  python eval_rl.py --mode pure --episodes 20 --print-episodes --render
  ```
- A more detailed example from README.md:
  ```bash
  python eval_rl.py --mode both --episodes 100 --out results_torque.csv
  ```
- eval_rl.py writes:
  - A per-episode CSV: for example results/eval_results_<seed>.csv.
  - A JSON summary: for example results/eval_results_<seed>_summary.json.

### 2.4. "Tests" and single-test runs

This repo does not use pytest or unittest; test files are executable scripts
meant to be run directly.

- Quick dependency sanity check (Gym + NumPy):
  ```bash
  python sanity_check.py
  ```
- Environment sanity / kinematics check (inverse kinematics to random targets):
  ```bash
  python test_env.py
  ```
- Visual tip/target debugging script (PyBullet overlays):
  ```bash
  python test_tip_visual.py
  ```

Running a single test

- To run a single scenario, run the corresponding script:
  - "Single test" == "single script" in this repo.
  - Example:
    ```bash
    python test_env.py
    ```
- If you introduce pytest later:
  - Place tests in test_*.py files.
  - Use:
    ```bash
    pytest -q
    pytest test_env.py::TestClassName::test_case_name
    ```
  - Update this document and README.md accordingly.

---

## 3. Code Style: General Principles

- Base style: PEP 8-like, but do not reformat entire files unnecessarily.
- Priority:
  1. Keep existing public behavior and CLI interfaces stable.
  2. Keep diffs small and focused.
  3. Match the surrounding style even when it is slightly non-standard.
- Comments and messages:
  - Existing comments and output are mostly in Portuguese; keep this consistent
    in new user-facing messages for now.
  - Internal helper comments can be in English if clearer.

---

## 4. Imports and Module Structure

- Order imports in three groups, separated by blank lines:
  1. Standard library: os, json, csv, time, argparse, random, etc.
  2. Third-party: numpy, gym, stable_baselines3, pybullet, etc.
  3. Local modules: from two_link_arm_env import TwoLinkArmEnv.
- Conventions:
  - import numpy as np
  - import pybullet as p
  - import gym
  - from stable_baselines3 import PPO
- Avoid duplicate imports (for example, import numpy as np twice in the same file).
- Prefer explicit imports over "from x import *".

---

## 5. Formatting and Structure

- Line length: aim for at most 100 characters; slightly longer is acceptable if it
  improves readability and avoids awkward breaks.
- Indentation: 4 spaces, no tabs.
- Spacing:
  - Surround binary operators with spaces: x ** 2, a + b, unless it harms
    clarity in mathematical formulas.
  - No trailing whitespace.
- Blank lines:
  - 2 blank lines before top-level def/class.
  - 1 blank line between logically separate code blocks inside functions.
- Functions and modules:
  - Keep files short and focused; this repo is small, so aim for clarity over
    heavy abstraction.
  - For new functionality, prefer adding helper functions instead of growing
    long monolithic functions.

---

## 6. Types and Interfaces

- Existing code is lightly typed:
  - Some functions use type hints (for example set_global_seed(seed: int)).
  - Most functions are untyped.
- Guidelines:
  - Adding type hints is welcome on new or refactored code, especially for
    public helpers and utilities.
  - Do not block on full typing; correctness and clarity come first.
  - Prefer standard typing (List, Tuple, Dict, Optional, Callable) or
    native generics on Python 3.9+.
- Keep function signatures stable for:
  - TwoLinkArmEnv methods.
  - CLI entrypoints: train_rl.py, eval_rl.py, main.py, test_*.py.

---

## 7. Naming Conventions

- Modules and files: snake_case.py (already followed).
- Classes: CamelCase (for example TwoLinkArmEnv).
- Functions and methods: snake_case (for example set_global_seed, _get_obs).
- Private helpers:
  - Prefix with _ inside classes (for example _apply_angles, _sample_target).
  - For module-level helpers that are not part of the public API, prefix with _
    only if there is a clear public API boundary.
- Constants: UPPER_SNAKE_CASE (for example EPISODES, STEPS).
- Variables:
  - Use descriptive names in non-performance-critical code.
  - Single-letter names are acceptable for math (for example x, y, D, tau1).

---

## 8. Error Handling and Logging

- Let truly unexpected errors propagate; do not blanket-catch exceptions unless
  there is a good reason.
- Expected errors:
  - Use explicit exceptions for file/model loading:
    - Example in eval_rl.py: raise FileNotFoundError when model is missing.
  - In control or math code, return None or a small sentinel when something is
    outside the feasible domain (for example target outside reachable workspace).
- Broad "except Exception":
  - Acceptable only around PyBullet GUI or debug overlays or other non-critical
    visualization code where failure should not kill the run (see
    test_tip_visual.py).
  - When adding such blocks, keep them tight and add a short comment describing
    why failures are ignored.
- Logging vs prints:
  - The existing code uses print for metrics and debug output; match this.
  - Do not introduce a logging framework unless explicitly requested.

---

## 9. Randomness and Reproducibility

- Use set_global_seed (see train_rl.py) when adding new experiment scripts.
- If you add new randomness sources (for example torch, new libraries), extend
  seeding accordingly.
- Do not silently change default seeds in existing CLIs without updating the
  documentation.

---

## 10. Gym and PyBullet Environment Conventions

- Gym version:
  - The environment follows the Gym 0.21 API (reset() -> obs,
    step(action) -> (obs, reward, done, info)).
  - If you add support for Gymnasium or newer APIs, keep backward-compatible
    shims or clearly separate new entrypoints.
- TwoLinkArmEnv:
  - Observation space: [theta1, theta2, target_x, target_y].
  - Action space: delta angles for each joint.
  - Reward modes:
    - "pure": distance plus light action penalty.
    - "pirl": adds torque-based effort penalty.
- Keep:
  - Joint indices, link indices, and offset handling consistent unless you fully
    update tests, training, and evaluation code.

---

## 11. Tooling, Linting, and Formatting

- There is currently no configured linter or formatter in the repo
  (.flake8, pyproject.toml tooling, .pre-commit-config.yaml etc. are
  absent).
- If you run tools locally (optional, not assumed):
  - ruff . or flake8 . for linting.
  - black . for formatting.
- When using automatic tools as an agent:
  - Avoid large format-only commits or sweeping style changes.
  - Prefer targeted fixes that address actual issues (bugs, clarity).

---

## 12. Cursor and Copilot Rules

- No Cursor rules (.cursor/rules/ or .cursorrules) are present.
- No Copilot instructions file (.github/copilot-instructions.md) is present.
- If such rules are added later, agents must:
  1. Read them.
  2. Update this AGENTS.md with a short summary of the key points.
  3. Follow those rules in addition to the guidelines above.

---
