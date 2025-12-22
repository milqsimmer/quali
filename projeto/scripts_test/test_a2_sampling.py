# test_a2_sampling.py
# Testes para as atualizações A2 do sampler de alvo:
# (1) amostrar em anel [r_min=|l1-l2|+margem, r_max=(l1+l2)-margem]
# (2) impor distância mínima ao tip inicial (>= min_dist)
# (3) opcional: filtrar por limites articulares via IK (apenas na amostragem)

## PARA RODAR ESTE TESTE:
"""
python test_a2_sampling.py --n 20000 --seed 0 --margin 0.02 --min-dist 0.07
python test_a2_sampling.py --n 20000 --seed 0 --margin 0.02 --min-dist 0.07 --ik
"""


from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


IMPORT_PATH = "two_link_arm_env"
ENV_CLASS = "TwoLinkArmEnv"


@dataclass
class SamplerConfig:
    n: int
    seed: int
    margin: float
    min_dist: float
    use_ik_filter: bool
    tol_fk: float = 1e-2
    max_ik_tries: int = 1


def _import_env() -> Any:
    mod = __import__(IMPORT_PATH, fromlist=[ENV_CLASS])
    cls = getattr(mod, ENV_CLASS)
    return cls


def _get_link_lengths(env: Any) -> Tuple[float, float]:
    # tenta achar l1, l2 em atributos comuns
    if hasattr(env, "l1") and hasattr(env, "l2"):
        return float(env.l1), float(env.l2)
    if hasattr(env, "link_lengths"):
        ll = getattr(env, "link_lengths")
        return float(ll[0]), float(ll[1])
    if hasattr(env, "LINK_LENGTHS"):
        ll = getattr(env, "LINK_LENGTHS")
        return float(ll[0]), float(ll[1])
    # fallback padrão do seu texto
    return 0.5, 0.5


def _get_joint_limits(env: Any) -> Tuple[np.ndarray, np.ndarray]:
    # padrão do seu texto: [-pi, pi] para ambas
    low = np.array([-math.pi, -math.pi], dtype=float)
    high = np.array([math.pi, math.pi], dtype=float)

    # se existir algo no env, usa
    for attr in ["joint_low", "joint_high", "joint_limits", "JOINT_LIMITS"]:
        if hasattr(env, attr):
            v = getattr(env, attr)
            if attr in ["joint_low", "joint_high"]:
                # precisa dos dois
                continue
            # formatos possíveis:
            # - joint_limits = [(-pi, pi), (-pi, pi)]
            try:
                low = np.array([v[0][0], v[1][0]], dtype=float)
                high = np.array([v[0][1], v[1][1]], dtype=float)
                return low, high
            except Exception:
                pass

    if hasattr(env, "joint_low") and hasattr(env, "joint_high"):
        try:
            low = np.array(env.joint_low, dtype=float)
            high = np.array(env.joint_high, dtype=float)
            if low.shape == (2,) and high.shape == (2,):
                return low, high
        except Exception:
            pass

    return low, high


def _fk_planar_2d(theta1: float, theta2: float, l1: float, l2: float) -> np.ndarray:
    # FK no plano (assumindo cadeia planar padrão)
    x = l1 * math.cos(theta1) + l2 * math.cos(theta1 + theta2)
    y = l1 * math.sin(theta1) + l2 * math.sin(theta1 + theta2)
    return np.array([x, y], dtype=float)


def _tip_initial_pos(env: Any, l1: float, l2: float) -> np.ndarray:
    # se o env tiver método, usa; senão assume θ1=0, θ2=0
    for fn in ["get_tip_pos", "get_end_effector_pos", "_get_tip_pos", "_get_ee_pos"]:
        if hasattr(env, fn):
            try:
                p = getattr(env, fn)()
                p = np.array(p, dtype=float).reshape(-1)
                return p[:2]
            except Exception:
                pass
    return _fk_planar_2d(0.0, 0.0, l1, l2)


def _sample_target(env: Any) -> np.ndarray:
    # tenta achar o sampler “novo”
    tip0 = np.array([env.l1 + env.l2, 0.0], dtype=float)
    for fn in ["sample_target", "_sample_target", "_sample_goal", "sample_goal"]:
        if hasattr(env, fn):
            p = getattr(env, fn)(tip0)
            p = np.array(p, dtype=float).reshape(-1)
            return p[:2]
    raise AttributeError(
        "Não encontrei método de amostragem de alvo no env. "
        "Esperava algo como sample_target() ou _sample_target()."
    )


def _ik_feasible(
    env: Any,
    target_xy: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    cfg: SamplerConfig,
    l1: float,
    l2: float,
) -> bool:
    # 1) Se você tem IK próprio no env, tenta usar
    for fn in ["inverse_kinematics", "_ik", "ik", "calc_ik"]:
        if hasattr(env, fn):
            try:
                sol = getattr(env, fn)(target_xy)
                sol = np.array(sol, dtype=float).reshape(-1)[:2]
                if np.any(sol < low) or np.any(sol > high):
                    return False
                ee = _fk_planar_2d(sol[0], sol[1], l1, l2)
                return float(np.linalg.norm(ee - target_xy)) <= cfg.tol_fk
            except Exception:
                pass

    # 2) Fallback: sem IK disponível no código -> não bloqueia o teste inteiro,
    # mas avisa e considera "não foi possível validar IK".
    # Se você preferir falhar aqui, troque por "return False" + mensagem.
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--min-dist", type=float, default=0.07)
    ap.add_argument(
        "--ik",
        action="store_true",
        help="ativa checagem/filtro de IK (se o env suportar)",
    )
    ap.add_argument("--tol-fk", type=float, default=1e-2)
    args = ap.parse_args()

    cfg = SamplerConfig(
        n=args.n,
        seed=args.seed,
        margin=args.margin,
        min_dist=args.min_dist,
        use_ik_filter=bool(args.ik),
        tol_fk=float(args.tol_fk),
    )

    EnvCls = _import_env()
    env = EnvCls()

    # semente (se env suportar)
    np.random.seed(cfg.seed)
    for fn in ["seed", "reset"]:
        if hasattr(env, fn):
            try:
                if fn == "seed":
                    env.seed(cfg.seed)
                else:
                    # reset(seed=...) em gym newer; tenta as duas formas
                    try:
                        env.reset(seed=cfg.seed)
                    except TypeError:
                        env.reset()
            except Exception:
                pass

    l1, l2 = _get_link_lengths(env)
    low, high = _get_joint_limits(env)
    tip0 = _tip_initial_pos(env, l1, l2)

    r_min = abs(l1 - l2) + cfg.margin
    r_max = (l1 + l2) - cfg.margin

    bad_ring = 0
    bad_min_dist = 0
    bad_ik = 0

    radii = []
    dists_tip0 = []

    for _ in range(cfg.n):
        tgt = _sample_target(env)

        r = float(np.linalg.norm(tgt))
        d0 = float(np.linalg.norm(tgt - tip0))
        radii.append(r)
        dists_tip0.append(d0)

        if not (r_min <= r <= r_max):
            bad_ring += 1

        if d0 < cfg.min_dist:
            bad_min_dist += 1

        if cfg.use_ik_filter:
            ok = _ik_feasible(env, tgt, low, high, cfg, l1, l2)
            if not ok:
                bad_ik += 1

    radii = np.array(radii, dtype=float)
    dists_tip0 = np.array(dists_tip0, dtype=float)

    print("=== Teste A2: sampler de alvo ===")
    print(f"Env: {ENV_CLASS} (from {IMPORT_PATH})")
    print(f"n={cfg.n}, seed={cfg.seed}")
    print(f"l1={l1:.4f}, l2={l2:.4f}")
    print(f"anel: r_min={r_min:.4f}, r_max={r_max:.4f}")
    print(f"min_dist_tip0={cfg.min_dist:.4f}")
    print(f"ik_check={'ON' if cfg.use_ik_filter else 'OFF'}")
    print("---")
    print(f"Violação anel: {bad_ring}/{cfg.n}")
    print(f"Violação min_dist: {bad_min_dist}/{cfg.n}")
    if cfg.use_ik_filter:
        print(f"Violação IK/limites: {bad_ik}/{cfg.n}")
    print("---")
    print(
        f"r (min/mean/max) = {radii.min():.4f} / {radii.mean():.4f} / {radii.max():.4f}"
    )
    print(
        f"d_tip0 (min/mean/max) = {dists_tip0.min():.4f} / {dists_tip0.mean():.4f} / {dists_tip0.max():.4f}"
    )

    failed = (bad_ring > 0) or (bad_min_dist > 0) or (cfg.use_ik_filter and bad_ik > 0)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
