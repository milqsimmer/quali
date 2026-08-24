# AGENTS guide for this repository

This file is for code agents (Cursor, Copilot, OpenAI, etc.) working in
`projeto/`, a reinforcement learning project for a 2D robotic arm using
Gym 0.21 and PyBullet.

Keep changes incremental, preserve existing behavior, and do not auto
reformat whole files.

## 1. Environment and setup

- Python version: project documentation says 3.8+, but code already uses
  builtin generics like `list[str]`, so assume at least Python 3.9.
- Dependencies: install from `requirements.txt` in this directory:

  ```bash
  python -m venv .venv
  source .venv/bin/activate  # on Windows: .venv\Scripts\activate
  pip install -r requirements.txt
  ```

- Core libraries: `gym==0.21`, `pybullet`, `numpy`, `stable_baselines3`,
  `matplotlib` and standard Python libs.
- Gym API is the legacy one: `reset() -> obs`, `step() -> (obs, reward,
  done, info)`. Do not switch to the newer Gymnasium API.

## 2. Build / lint / test commands

There is no packaging or build pipeline; scripts are run directly.

### Build

- No dedicated build step. For a quick syntax check on all modules:

  ```bash
  python -m compileall .
  ```

### Lint and formatting

- No project linter or formatter config files are present (no `flake8`,
  `ruff`, `black` configs detected).
- When editing, keep formatting consistent with surrounding code and
  avoid running whole-file autoformatters unless explicitly requested.
- If you must format, limit it to the lines you touch.

### Tests and quick checks

There is no `pytest` configuration; tests are small scripts under
`scripts_test/`.

- Sanity check of Gym + NumPy:

  ```bash
  python scripts_test/sanity_check.py
  ```

- Inverse kinematics + environment distance check (closest thing to a
  single focused test):

  ```bash
  python scripts_test/test_env.py
  ```

- Visual debug of tip vs target in the PyBullet GUI:

  ```bash
  python scripts_test/test_tip_visual.py
  ```

- Demo of the original PyBullet simulation (no RL):

  ```bash
  python scripts_test/demo_inicial.py
  ```

If you want to add formal unit tests, prefer `pytest`, put them under a
`tests/` directory, and keep the existing scripts working as-is.

## 3. Running experiments

- Single PPO training run:

  ```bash
  python train_rl.py --mode pure --seed 0 --steps 300000
  python train_rl.py --mode pirl --seed 0 --steps 300000
  ```

- Evaluation of trained models:

  ```bash
  python eval_rl.py --mode pure --episodes 20 --print-episodes --render
  python eval_rl.py --mode pirl --episodes 100
  python eval_rl.py --mode both --episodes 100 --out results/eval_results --seed 0
  ```

- Sweep over seeds for training and evaluation:

  ```bash
  python run_experiments_seeds.py --first-seed 0 --last-seed 4
  ```

- Analysis of scales (distance, action norm, torque, energy):

  ```bash
  python analyze_scales.py --mode both --policy trained
  ```

- Detailed per-step inspection and CSV dumps:

  ```bash
  python inspect_episodes.py --mode both --first-seed 0 --last-seed 4 \
      --episodes-per-config 20 --max-steps 200
  ```

## 4. Code layout

- `two_link_arm_env.py`: Gym environment `TwoLinkArmEnv` built on
  PyBullet, with reward modes `pure` and `pirl` and detailed info
  fields in `info`.
- `train_rl.py`: CLI that trains PPO policies and saves monitor CSVs
  and model zip files under `runs_pure/` and `runs_pirl/`.
- `eval_rl.py`: CLI that loads trained models, runs evaluation episodes,
  prints a summary table, and writes CSV plus JSON summary files.
- `analyze_scales.py`: runs policies (random or trained) to measure
  typical scales for distance, actions, torque and energy, and produces
  bar plots.
- `inspect_episodes.py`: more detailed rollouts with rich per-step
  metrics and trajectory CSVs.
- `run_experiments_seeds.py`: orchestrates training + evaluation over a
  range of seeds by spawning `train_rl.py` and `eval_rl.py` as
  subprocesses.
- `scripts_test/`: small scripts used as sanity tests and visual debug.

## 5. Python style guidelines

### Imports

- Group imports as:
  1) standard library (`os`, `sys`, `argparse`, `time`, `json`, etc.),
  2) third-party (`numpy`, `gym`, `pybullet`, `stable_baselines3`,
     `matplotlib`),
  3) local (`from two_link_arm_env import TwoLinkArmEnv`).
- Within a group, keep imports alphabetic where reasonable.
- Prefer `from module import Name` for local modules used widely inside
  a file.

### Formatting

- Indentation is 4 spaces; no tabs.
- Aim for Black-like formatting but do not reformat entire files.
- Break long argument lists over multiple lines with trailing commas.
- Use double quotes for docstrings; string literal style inside code can
  follow the existing file (mixed `'` and `"` is acceptable, do not
  churn them).

### Types

- Use standard type hints for public functions and CLIs where it adds
  clarity (examples already exist in `run_experiments_seeds.py` and
  `inspect_episodes.py`).
- Prefer builtin generics (`list[str]`, `dict[str, float]`) to
  `List[str]` unless you need compatibility with type-checkers that do
  not support them.
- For numpy arrays, use `np.ndarray` or `np.ndarray[Any, Any]` if you
  need explicit types; otherwise keep hints simple.

### Naming

- Modules and functions: `snake_case`.
- Variables: `snake_case`, descriptive but concise (`ep_ret`,
  `tau_sum`, `tip_x`).
- Classes: `CapWords` (`TwoLinkArmEnv`).
- Constants: `UPPER_SNAKE_CASE` for file-local constants
  (`EPISODES`, `STEPS`, etc.).

### Functions and modules

- Top-level scripts follow a `main()` function plus an
  `if __name__ == "__main__": main()` guard.
- Use docstrings at the module or function level when behavior is
  nontrivial, especially for scripts and analysis tools.
- Keep side effects (file system writes, heavy training loops) behind a
  function or CLI entry point.

### Error handling and logging

- For missing models or files, raise `FileNotFoundError` with a clear
  message (see `eval_rl.py`, `inspect_episodes.py`).
- For invalid CLI arguments that pass argparse but fail logical checks
  (like `first-seed > last-seed`), exit with `SystemExit` and a short
  explanation.
- Use `try/except` sparingly; most current uses are around optional
  debug overlays or best-effort metrics. Follow that pattern: swallow
  only noncritical errors.
- Logging is done with `print` in scripts; follow the existing style and
  keep messages in Portuguese where the surrounding file does.

## 6. RL and environment specifics

- The info dict returned by `TwoLinkArmEnv.step` already contains
  several fields (`distance`, `final_distance`, `tau_sum_total`,
  `episode_energy`, etc.). When extending the environment, prefer adding
  new metrics here rather than changing existing keys.
- Reward structure differs by `reward_mode` (`pure` vs `pirl`). If you
  modify rewards, keep backward compatibility where possible and update
  comments in `two_link_arm_env.py` and relevant scripts.
- Torque limits and energy accounting are important for later analysis;
  do not silently change units or conventions without updating plotting
  and inspection scripts.

## 7. Cursor and Copilot rules

- As of this version there are no `.cursor/rules/`, `.cursorrules`, or
  `.github/copilot-instructions.md` files in the repository.
- If you introduce such rules, update this section with a short summary
  so other agents know which constraints are enforced.

## 8. When adding new code

- Mirror the existing structure: small, focused scripts with clear
  CLIs; keep long numerical experiments out of library-style modules.
- Keep backward compatibility with existing command line interfaces
  where reasonable; add new flags rather than renaming old ones.
- Prefer adding new analysis scripts next to the existing ones in this
  directory or under `scripts_test/` for quick sanity checks.
