import os
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, FancyArrowPatch


ENV_FILE = "two_link_arm_env.py"
URDF_FILE = "two_link_arm.urdf"


def load_env_class(env_file_path):
    spec = importlib.util.spec_from_file_location(
        "two_link_arm_env_module", env_file_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TwoLinkArmEnv


def forward_kinematics(theta1, theta2, l1, l2):
    """Calcula a cinemática direta do manipulador planar de 2 elos:
    - O: origem (base) no ponto (0, 0).
    - P: posição da junta intermediária (fim do elo 1).
    - Q: posição do efetuador final (fim do elo 2).
    """
    O = np.array([0.0, 0.0])
    P = np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    Q = np.array(
        [
            l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2),
            l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2),
        ]
    )
    return O, P, Q


def inverse_kinematics_2link(x, y, l1, l2, elbow_up=True):
    """Resolve a cinemática inversa para um manipulador planar de 2 elos:
    dado um ponto alvo (x, y) e comprimentos l1, l2, calcula os ângulos theta1 e theta2.
    """

    r2 = x**2 + y**2
    c2 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    c2 = np.clip(c2, -1.0, 1.0)

    if elbow_up:
        s2 = np.sqrt(1 - c2**2)
    else:
        s2 = -np.sqrt(1 - c2**2)

    theta2 = np.arctan2(s2, c2)
    theta1 = np.arctan2(y, x) - np.arctan2(l2 * s2, l1 + l2 * c2)
    return theta1, theta2


def add_label(ax, x, y, text, fontsize=16, ha="center", va="center"):
    """
    Escreve texto com fundo branco semi-transparente para melhorar legibilidade.
    """
    ax.text(
        x,
        y,
        text,
        fontsize=fontsize,
        ha=ha,
        va=va,
        zorder=11,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.2),
    )


def place_label_on_link(start, end, offset=0.06):
    """
    Posição de legenda deslocada perpendicularmente ao elo.
    Boa para l1, l2 e também para nomes próximos de segmentos.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)

    mid = (start + end) / 2.0
    direction = end - start
    norm = np.linalg.norm(direction)

    if norm < 1e-9:
        return mid

    direction = direction / norm
    normal = np.array([-direction[1], direction[0]])

    # força a legenda preferencialmente para cima
    if normal[1] < 0:
        normal = -normal

    return mid + offset * normal


def place_label_away_from_point(point, avoid_point, offset=0.06):
    """
    Coloca a legenda afastando-a de um ponto que queremos evitar.
    Boa para target, Q, P etc.
    """
    point = np.asarray(point, dtype=float)
    avoid_point = np.asarray(avoid_point, dtype=float)

    direction = point - avoid_point
    norm = np.linalg.norm(direction)

    if norm < 1e-9:
        return point + np.array([offset, offset])

    direction = direction / norm
    return point + offset * direction


def place_angle_label(
    center, radius, angle_start_deg, angle_end_deg, extra_offset=0.05
):
    """
    Coloca o texto do ângulo ao longo da bissetriz do arco.
    """
    angle_mid_deg = 0.5 * (angle_start_deg + angle_end_deg)
    angle_mid = np.radians(angle_mid_deg)

    return np.array(
        [
            center[0] + (radius + extra_offset) * np.cos(angle_mid),
            center[1] + (radius + extra_offset) * np.sin(angle_mid),
        ]
    )


def draw_arrow(ax, start, end, label=None, text_offset=(0.0, 0.0), lw=1.5):
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=lw,
        color="black",
        zorder=10,
    )
    ax.add_patch(arrow)

    if label is not None:
        mid = (np.array(start) + np.array(end)) / 2.0
        add_label(
            ax, mid[0] + text_offset[0], mid[1] + text_offset[1], label, fontsize=16
        )


def draw_joint(ax, center, r_outer=0.03, r_inner=0.012):
    outer = Circle(
        center, r_outer, facecolor="white", edgecolor="black", linewidth=2.0, zorder=5
    )
    inner = Circle(
        center, r_inner, facecolor="white", edgecolor="black", linewidth=2.0, zorder=6
    )
    ax.add_patch(outer)
    ax.add_patch(inner)


def draw_local_frame(ax, origin, angle, label_x, label_y, scale=0.14):
    ex = np.array([np.cos(angle), np.sin(angle)])
    ey = np.array([-np.sin(angle), np.cos(angle)])

    x_end = origin + scale * ex
    y_end = origin + scale * ey

    # desenha setas sem label
    draw_arrow(ax, origin, x_end, label=None, lw=1.1)
    draw_arrow(ax, origin, y_end, label=None, lw=1.1)

    # posição do texto: desloca para ESQUERDA da seta
    normal_x = np.array([-ex[1], ex[0]])  # perpendicular ao eixo x local
    normal_y = np.array([-ey[1], ey[0]])  # perpendicular ao eixo y local

    # força sempre para esquerda (lado negativo)
    normal_x = -normal_x
    normal_y = -normal_y

    label_x_pos = x_end + 0.04 * normal_x
    label_y_pos = y_end + 0.04 * normal_y

    add_label(ax, label_x_pos[0], label_x_pos[1], label_x, fontsize=14)
    add_label(ax, label_y_pos[0], label_y_pos[1], label_y, fontsize=14)


def draw_angle_arc(
    ax, center, radius, angle_start_deg, angle_end_deg, label, label_pos
):
    arc = Arc(
        center,
        2 * radius,
        2 * radius,
        angle=0,
        theta1=angle_start_deg,
        theta2=angle_end_deg,
        linewidth=1.5,
        color="black",
    )
    ax.add_patch(arc)
    add_label(ax, label_pos[0], label_pos[1], label, fontsize=16)


def draw_gripper(ax, Q, angle):
    tip_dir = np.array([np.cos(angle), np.sin(angle)])
    tip_perp = np.array([-np.sin(angle), np.cos(angle)])

    wrist = Q - 0.015 * tip_dir
    claw_base_1 = wrist + 0.02 * tip_perp
    claw_base_2 = wrist - 0.02 * tip_perp
    claw_tip_1 = claw_base_1 + 0.05 * tip_dir + 0.012 * tip_perp
    claw_tip_2 = claw_base_2 + 0.05 * tip_dir - 0.012 * tip_perp

    ax.plot(
        [claw_base_1[0], claw_tip_1[0]],
        [claw_base_1[1], claw_tip_1[1]],
        color="black",
        linewidth=1.4,
    )
    ax.plot(
        [claw_base_2[0], claw_tip_2[0]],
        [claw_base_2[1], claw_tip_2[1]],
        color="black",
        linewidth=1.4,
    )


def plot_arm(ax, theta1, theta2, l1, l2, target=None, title=None):
    LINK1_COLOR = "#1f77b4"
    LINK2_COLOR = "#ff7f0e"
    LINK_WIDTH = 6

    O, P, Q = forward_kinematics(theta1, theta2, l1, l2)

    draw_arrow(ax, (-0.02, 0.0), (1.22, 0.0), lw=1.5)
    draw_arrow(ax, (0.0, -0.02), (0.0, 1.22), lw=1.5)

    ax.plot(
        [O[0], P[0]],
        [O[1], P[1]],
        color=LINK1_COLOR,
        linewidth=LINK_WIDTH,
        solid_capstyle="round",
        zorder=2,
    )

    ax.plot(
        [P[0], Q[0]],
        [P[1], Q[1]],
        color=LINK2_COLOR,
        linewidth=LINK_WIDTH,
        solid_capstyle="round",
        zorder=2,
    )

    draw_joint(ax, O)
    draw_joint(ax, P)
    draw_gripper(ax, Q, theta1 + theta2)

    pos_O = place_label_away_from_point(O, P, offset=0.08)
    pos_P = place_label_away_from_point(P, (O + Q) / 2.0, offset=0.07)
    pos_Q = place_label_away_from_point(Q, P, offset=0.07)

    add_label(ax, pos_O[0], pos_O[1], "O", fontsize=16)
    add_label(ax, pos_P[0], pos_P[1], "P", fontsize=16)
    add_label(ax, pos_Q[0], pos_Q[1], "Q", fontsize=16)

    pos_l1 = place_label_on_link(O, P, offset=0.06)
    pos_l2 = place_label_on_link(P, Q, offset=0.06)

    ax.text(
        pos_l1[0],
        pos_l1[1],
        r"$l_1$",
        fontsize=18,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )

    ax.text(
        pos_l2[0],
        pos_l2[1],
        r"$l_2$",
        fontsize=18,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
    )

    draw_local_frame(ax, P, theta1, r"$x_1$", r"$y_1$", scale=0.13)
    draw_local_frame(ax, Q, theta1 + theta2, r"$x_2$", r"$y_2$", scale=0.13)

    theta1_deg = np.degrees(theta1)
    if abs(theta1_deg) > 1e-4:
        label_theta1 = place_angle_label(
            center=O,
            radius=0.14,
            angle_start_deg=0,
            angle_end_deg=theta1_deg,
            extra_offset=0.05,
        )

        draw_angle_arc(
            ax=ax,
            center=O,
            radius=0.14,
            angle_start_deg=0,
            angle_end_deg=theta1_deg,
            label=r"$\theta_1$",
            label_pos=label_theta1,
        )

    theta2_deg = np.degrees(theta2)
    if abs(theta2_deg) > 1e-4:
        label_theta2 = place_angle_label(
            center=P,
            radius=0.11,
            angle_start_deg=theta1_deg,
            angle_end_deg=theta1_deg + theta2_deg,
            extra_offset=0.05,
        )

        draw_angle_arc(
            ax=ax,
            center=P,
            radius=0.11,
            angle_start_deg=theta1_deg,
            angle_end_deg=theta1_deg + theta2_deg,
            label=r"$\theta_2$",
            label_pos=label_theta2,
        )

    if target is not None:
        ax.plot(target[0], target[1], marker="o", markersize=10, color="red", mew=2)

        pos_target = place_label_away_from_point(target, Q, offset=0.08)
        add_label(ax, pos_target[0], pos_target[1], "target", fontsize=14)

    if title:
        ax.set_title(title, fontsize=14)

    limit = l1 + l2 + 0.2

    ax.set_xlim(-0.4, limit)
    ax.set_ylim(-limit, limit)

    ax.set_xticks(np.arange(-0.4, limit, 0.1))
    ax.set_yticks(np.arange(-limit, limit, 0.1))

    ax.grid(True, which="both", linewidth=0.3, linestyle="--", alpha=0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)

    ax.tick_params(axis="both", which="major", labelsize=8)

    ax.spines["left"].set_position("zero")
    ax.spines["bottom"].set_position("zero")

    ax.set_aspect("equal")
    ax.axis("on")

    ax.text(ax.get_xlim()[1] + 0.1, 0, "X", fontsize=12, ha="right")
    ax.text(0, ax.get_ylim()[1] + 0.1, "Y", fontsize=12, va="top")


def main():
    if not os.path.exists(ENV_FILE):
        raise FileNotFoundError(f"Arquivo não encontrado: {ENV_FILE}")

    if not os.path.exists(URDF_FILE):
        raise FileNotFoundError(f"Arquivo não encontrado: {URDF_FILE}")

    TwoLinkArmEnv = load_env_class(ENV_FILE)
    env = TwoLinkArmEnv(render=False)

    try:
        l1 = env.l1
        l2 = env.l2

        # usa o próprio env para gerar alvo e estado inicial
        obs = env.reset()
        theta1_init = float(obs[0])
        theta2_init = float(obs[1])
        target = np.array([float(obs[2]), float(obs[3])])

        print(f"l1 = {l1}")
        print(f"l2 = {l2}")
        print(f"theta1_init = {theta1_init}")
        print(f"theta2_init = {theta2_init}")
        print(f"target sampled by env = ({target[0]:.6f}, {target[1]:.6f})")

        # figura 1: posição inicial
        fig1, ax1 = plt.subplots(figsize=(6, 6))
        plot_arm(
            ax=ax1,
            theta1=theta1_init,
            theta2=theta2_init,
            l1=l1,
            l2=l2,
            target=None,
        )
        fig1.tight_layout()
        fig1.savefig("arm_initial_from_env.png", dpi=300, bbox_inches="tight")
        fig1.savefig("arm_initial_from_env.svg", bbox_inches="tight")

        # # figura 2: alcançando o alvo gerado pelo env
        # theta1_goal, theta2_goal = inverse_kinematics_2link(
        #     x=target[0], y=target[1], l1=l1, l2=l2, elbow_up=True
        # )

        # fig2, ax2 = plt.subplots(figsize=(6, 6))
        # plot_arm(
        #     ax=ax2,
        #     theta1=theta1_goal,
        #     theta2=theta2_goal,
        #     l1=l1,
        #     l2=l2,
        #     target=target,
        # )
        # fig2.tight_layout()
        # fig2.savefig("arm_target_sampled_from_env.png", dpi=300, bbox_inches="tight")
        # fig2.savefig("arm_target_sampled_from_env.svg", bbox_inches="tight")

        plt.show()

        # print("\nArquivos gerados:")
        # print("- arm_initial_from_env.png")
        # print("- arm_initial_from_env.svg")
        # print("- arm_target_sampled_from_env.png")
        # print("- arm_target_sampled_from_env.svg")

        # print("\nÂngulos para o alvo amostrado:")
        # print(f"theta1 = {theta1_goal:.6f} rad ({np.degrees(theta1_goal):.2f} deg)")
        # print(f"theta2 = {theta2_goal:.6f} rad ({np.degrees(theta2_goal):.2f} deg)")

    finally:
        print("\nFechando o ambiente...")
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
