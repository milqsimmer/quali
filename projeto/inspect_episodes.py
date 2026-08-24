"""Ferramenta de inspeção detalhada de episódios.

Este script carrega modelos treinados (pure/pirl) e executa vários
episódios adicionais por (modo, seed), registrando passo a passo:
  - estados (theta1, theta2, alvo), ações e recompensas,
  - distância ponta–alvo, torques, velocidades, potência, energia,
  - posição real da ponta (tip_x, tip_y) e saturação de torque.

Para cada (modo, seed), ele seleciona episódios "representativos"
(sucesso de baixa/alta energia e falha de alta energia) e salva
CSV detalhados em results/episodios_inspecao/.

Uso recomendado (exemplo):

    python inspect_episodes.py --mode both --first-seed 0 --last-seed 4 \
        --episodes-per-config 20 --max-steps 200

"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3 import PPO

from two_link_arm_env import TwoLinkArmEnv


def get_model_path(mode: str, seed: int) -> str:
    """Constroi o caminho padrao do modelo treinado."""
    run_dir = os.path.join(f"runs_{mode}", f"seed_{seed}")
    return os.path.join(run_dir, f"ppo_model_{seed}.zip")


def rollout_episodes(
    mode: str,
    seed: int,
    episodes: int,
    max_steps: int,
) -> List[Dict[str, object]]:
    """Executa varios episodios e retorna lista de resultados.

    Cada item contem:
      - "summary": dict com metricas por episodio
      - "traj": dict com listas passo a passo (para salvar depois)
    """

    model_path = get_model_path(mode, seed)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modelo nao encontrado: {model_path}")

    env = TwoLinkArmEnv(render=False, reward_mode=mode)
    env = TimeLimit(env, max_episode_steps=max_steps)
    model = PPO.load(model_path, env=env)

    # tenta obter dt do ambiente base
    base_env = env
    if hasattr(base_env, "env"):
        base_env = base_env.env
    dt = getattr(base_env, "dt", 1.0 / 240.0)

    # limite de torque (forca) por junta usado no ambiente
    torque_max = 20.0
    sat_tol = 0.99  # considera saturado a partir de 99% do maximo

    all_eps: List[Dict[str, object]] = []

    for ep in range(episodes):
        obs = env.reset()
        done = False

        # distancia inicial do alvo em relacao à base
        th1_0, th2_0, tx0, ty0 = map(float, obs)
        target_r0 = float(np.hypot(tx0, ty0))

        step_idx: List[int] = []
        theta1: List[float] = []
        theta2: List[float] = []
        target_x: List[float] = []
        target_y: List[float] = []
        act0: List[float] = []
        act1: List[float] = []
        distances: List[float] = []
        tau1_list: List[float] = []
        tau2_list: List[float] = []
        tau_sum_list: List[float] = []
        omega1_list: List[float] = []
        omega2_list: List[float] = []
        power_list: List[float] = []
        cum_energy_list: List[float] = []
        tip_x_list: List[float] = []
        tip_y_list: List[float] = []
        sat1_list: List[bool] = []
        sat2_list: List[bool] = []
        sat_any_list: List[bool] = []
        rewards: List[float] = []
        dones: List[bool] = []

        ep_ret = 0.0
        ep_len = 0
        ep_tau_sum_total = 0.0
        ep_energy = 0.0
        sat1_count = 0
        sat2_count = 0
        sat_any_count = 0

        # registra estado inicial (passo 0) incluindo posicao da ponta
        try:
            ee0 = base_env._get_end_effector_pos()  # type: ignore[attr-defined]
            tip_x0, tip_y0 = float(ee0[0]), float(ee0[1])
        except Exception:
            tip_x0 = tip_y0 = float("nan")

        step_idx.append(0)
        theta1.append(th1_0)
        theta2.append(th2_0)
        target_x.append(tx0)
        target_y.append(ty0)
        act0.append(0.0)
        act1.append(0.0)

        # distancia inicial ponta-alvo
        if np.isfinite(tip_x0) and np.isfinite(tip_y0):
            d0 = float(np.hypot(tip_x0 - tx0, tip_y0 - ty0))
        else:
            d0 = float("nan")

        distances.append(d0)
        tau1_list.append(0.0)
        tau2_list.append(0.0)
        tau_sum_list.append(0.0)
        omega1_list.append(0.0)
        omega2_list.append(0.0)
        power_list.append(0.0)
        cum_energy_list.append(0.0)
        tip_x_list.append(tip_x0)
        tip_y_list.append(tip_y0)
        sat1_list.append(False)
        sat2_list.append(False)
        sat_any_list.append(False)
        rewards.append(0.0)
        dones.append(False)

        while not done:
            # obs = [theta1, theta2, target_x, target_y]
            th1, th2, tx, ty = map(float, obs)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            ep_ret += float(reward)
            ep_len += 1

            dist = float(info.get("distance", np.nan))
            tau1 = float(info.get("tau1", np.nan))
            tau2 = float(info.get("tau2", np.nan))
            tau_sum = float(info.get("tau_sum", np.nan))
            omega1 = float(info.get("omega1", np.nan))
            omega2 = float(info.get("omega2", np.nan))
            power = float(info.get("power", np.nan))

            if not np.isnan(tau_sum):
                ep_tau_sum_total += tau_sum
            if not np.isnan(power):
                ep_energy += power * dt

            # saturacao de torque por junta
            sat1 = not np.isnan(tau1) and abs(tau1) >= sat_tol * torque_max
            sat2 = not np.isnan(tau2) and abs(tau2) >= sat_tol * torque_max
            sat_any = sat1 or sat2

            if sat1:
                sat1_count += 1
            if sat2:
                sat2_count += 1
            if sat_any:
                sat_any_count += 1

            step_idx.append(ep_len)
            theta1.append(th1)
            theta2.append(th2)
            target_x.append(tx)
            target_y.append(ty)
            act0.append(float(action[0]))
            act1.append(float(action[1]))
            distances.append(dist)
            tau1_list.append(tau1)
            tau2_list.append(tau2)
            tau_sum_list.append(tau_sum)
            omega1_list.append(omega1)
            omega2_list.append(omega2)
            power_list.append(power)
            cum_energy_list.append(ep_energy)
            sat1_list.append(sat1)
            sat2_list.append(sat2)
            sat_any_list.append(sat_any)
            rewards.append(float(reward))
            dones.append(bool(done))

            # posicao da ponta (tip) via ambiente base, se disponivel
            try:
                ee_pos = base_env._get_end_effector_pos()  # type: ignore[attr-defined]
                tip_x_list.append(float(ee_pos[0]))
                tip_y_list.append(float(ee_pos[1]))
            except Exception:
                tip_x_list.append(float("nan"))
                tip_y_list.append(float("nan"))

        final_dist = float(info.get("distance", np.nan))
        success = int(final_dist < 0.05)

        if ep_len > 0:
            tau_effort = ep_tau_sum_total / ep_len
            sat1_rate = sat1_count / ep_len
            sat2_rate = sat2_count / ep_len
            sat_any_rate = sat_any_count / ep_len
        else:
            tau_effort = float("nan")
            sat1_rate = sat2_rate = sat_any_rate = float("nan")

        summary = {
            "mode": mode,
            "seed": seed,
            "episode_idx": ep,
            "success": success,
            "final_distance": final_dist,
            "length": ep_len,
            "return": ep_ret,
            "tau_effort": tau_effort,
            "tau_sum_total": ep_tau_sum_total,
            "energy": ep_energy,
            "target_radius": target_r0,
            "torque_sat1_count": sat1_count,
            "torque_sat2_count": sat2_count,
            "torque_sat_any_count": sat_any_count,
            "torque_sat1_rate": sat1_rate,
            "torque_sat2_rate": sat2_rate,
            "torque_sat_any_rate": sat_any_rate,
        }

        traj = {
            "step": step_idx,
            "theta1": theta1,
            "theta2": theta2,
            "target_x": target_x,
            "target_y": target_y,
            "action0": act0,
            "action1": act1,
            "distance": distances,
            "tau1": tau1_list,
            "tau2": tau2_list,
            "tau_sum": tau_sum_list,
            "omega1": omega1_list,
            "omega2": omega2_list,
            "power": power_list,
            "cumulative_energy": cum_energy_list,
            "tip_x": tip_x_list,
            "tip_y": tip_y_list,
            "sat1": sat1_list,
            "sat2": sat2_list,
            "sat_any": sat_any_list,
            "reward": rewards,
            "done": dones,
        }

        all_eps.append({"summary": summary, "traj": traj})

    env.close()
    return all_eps


def pick_representative_episodes(
    episodes: List[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """Seleciona episodios representativos a partir de um lote.

    Retorna um dicionario tag->episodio, onde tag pode ser:
      - "low_energy_success"
      - "high_energy_success"
      - "high_energy_fail"
    """

    # extrai listas de indices e summaries
    summaries: List[Dict[str, object]] = [e["summary"] for e in episodes]  # type: ignore[index]

    def indices_where(cond) -> List[int]:
        return [i for i, s in enumerate(summaries) if cond(s)]

    idx_succ = indices_where(lambda s: s["success"] == 1)
    idx_fail = indices_where(lambda s: s["success"] == 0)

    selected: Dict[str, Dict[str, object]] = {}

    def best_by_energy(indices: List[int], *, reverse: bool) -> int | None:
        if not indices:
            return None
        energies = [summaries[i]["energy"] for i in indices]
        order = np.argsort(energies)
        if reverse:
            return indices[int(order[-1])]
        return indices[int(order[0])]

    # sucesso com menor energia
    i_low = best_by_energy(idx_succ, reverse=False)
    if i_low is not None:
        selected["low_energy_success"] = episodes[i_low]

    # sucesso com maior energia
    i_high = best_by_energy(idx_succ, reverse=True)
    if i_high is not None:
        selected["high_energy_success"] = episodes[i_high]

    # falha com maior energia
    i_fail_high = best_by_energy(idx_fail, reverse=True)
    if i_fail_high is not None:
        selected["high_energy_fail"] = episodes[i_fail_high]

    return selected


def save_episode_csv(
    out_path: str,
    summary: Dict[str, object],
    traj: Dict[str, List[object]],
) -> None:
    import csv

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # escreve um CSV com colunas de resumo + trajetoria passo a passo
    fieldnames = [
        "mode",
        "seed",
        "episode_idx",
        "success",
        "final_distance",
        "length",
        "return",
        "tau_effort",
        "tau_sum_total",
        "energy",
        "target_radius",
        "torque_sat1_count",
        "torque_sat2_count",
        "torque_sat_any_count",
        "torque_sat1_rate",
        "torque_sat2_rate",
        "torque_sat_any_rate",
        "step",
        "theta1",
        "theta2",
        "target_x",
        "target_y",
        "action0",
        "action1",
        "distance",
        "tau1",
        "tau2",
        "tau_sum",
        "omega1",
        "omega2",
        "power",
        "cumulative_energy",
        "tip_x",
        "tip_y",
        "sat1",
        "sat2",
        "sat_any",
        "reward",
        "done",
    ]

    # numero de passos
    n_steps = len(traj["step"])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(n_steps):
            row = {
                "mode": summary["mode"],
                "seed": summary["seed"],
                "episode_idx": summary["episode_idx"],
                "success": summary["success"],
                "final_distance": summary["final_distance"],
                "length": summary["length"],
                "return": summary["return"],
                "tau_effort": summary["tau_effort"],
                "tau_sum_total": summary["tau_sum_total"],
                "energy": summary["energy"],
                "target_radius": summary["target_radius"],
                "torque_sat1_count": summary["torque_sat1_count"],
                "torque_sat2_count": summary["torque_sat2_count"],
                "torque_sat_any_count": summary["torque_sat_any_count"],
                "torque_sat1_rate": summary["torque_sat1_rate"],
                "torque_sat2_rate": summary["torque_sat2_rate"],
                "torque_sat_any_rate": summary["torque_sat_any_rate"],
                "step": traj["step"][i],
                "theta1": traj["theta1"][i],
                "theta2": traj["theta2"][i],
                "target_x": traj["target_x"][i],
                "target_y": traj["target_y"][i],
                "action0": traj["action0"][i],
                "action1": traj["action1"][i],
                "distance": traj["distance"][i],
                "tau1": traj["tau1"][i],
                "tau2": traj["tau2"][i],
                "tau_sum": traj["tau_sum"][i],
                "omega1": traj["omega1"][i],
                "omega2": traj["omega2"][i],
                "power": traj["power"][i],
                "cumulative_energy": traj["cumulative_energy"][i],
                "tip_x": traj["tip_x"][i],
                "tip_y": traj["tip_y"][i],
                "sat1": traj["sat1"][i],
                "sat2": traj["sat2"][i],
                "sat_any": traj["sat_any"][i],
                "reward": traj["reward"][i],
                "done": traj["done"][i],
            }
            writer.writerow(row)


def parse_seeds_arg(
    seeds_str: str | None, first: int | None, last: int | None
) -> List[int]:
    if seeds_str:
        return [int(s.strip()) for s in seeds_str.split(",") if s.strip()]
    if first is not None and last is not None:
        if first > last:
            raise SystemExit("first-seed deve ser <= last-seed")
        return list(range(first, last + 1))
    raise SystemExit("Informe --seeds ou (--first-seed e --last-seed)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Executa episodios adicionais para inspecao detalhada, "
            "gravando trajetorias passo a passo para alguns episodios "
            "representativos (baixa/alta energia, sucesso/fracasso)."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["pure", "pirl", "both"],
        default="both",
        help="Qual modo inspecionar (pure, pirl ou both).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Lista de seeds (ex.: '0,1,2,3,4').",
    )
    parser.add_argument(
        "--first-seed",
        type=int,
        default=None,
        help="Seed inicial (alternativa a --seeds).",
    )
    parser.add_argument(
        "--last-seed",
        type=int,
        default=None,
        help="Seed final (alternativa a --seeds).",
    )
    parser.add_argument(
        "--episodes-per-config",
        type=int,
        default=20,
        help="Numero de episodios simulados por (modo, seed).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Limite de passos por episodio (TimeLimit).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/episodios_inspecao",
        help="Diretorio base para salvar episodios inspecionados.",
    )

    args = parser.parse_args()

    seeds = parse_seeds_arg(args.seeds, args.first_seed, args.last_seed)
    modes = ["pure", "pirl"] if args.mode == "both" else [args.mode]

    for seed in seeds:
        for mode in modes:
            print("\n" + "=" * 80)
            print(f"Inspecionando modo={mode}, seed={seed}")
            print("=" * 80)

            episodes = rollout_episodes(
                mode=mode,
                seed=seed,
                episodes=args.episodes_per_config,
                max_steps=args.max_steps,
            )

            selected = pick_representative_episodes(episodes)

            for tag, ep_data in selected.items():
                summary = ep_data["summary"]  # type: ignore[index]
                traj = ep_data["traj"]  # type: ignore[index]

                fname = f"mode_{mode}_seed{seed}_{tag}.csv"
                out_path = os.path.join(args.out_dir, fname)
                print(f"  Salvando episodio '{tag}' em: {out_path}")
                save_episode_csv(out_path, summary, traj)


if __name__ == "__main__":
    main()
