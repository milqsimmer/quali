import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc, FancyArrowPatch

# =========================
# Configuração do manipulador
# =========================
L1 = 0.5
L2 = 0.5

# alvo desejado
TARGET = np.array([0.7, 0.6])

# posição inicial do seu ambiente
THETA1_INIT = 0.0
THETA2_INIT = 0.0


# =========================
# Cinemática
# =========================
def forward_kinematics(theta1, theta2, l1=L1, l2=L2):
    """
    Retorna:
    O = origem/base
    P = junta intermediária
    Q = ponta (tip)
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


def inverse_kinematics_2link(x, y, l1=L1, l2=L2, elbow_up=True):
    """
    Resolve a cinemática inversa de um braço planar 2R.

    elbow_up=True  -> solução "cotovelo para cima"
    elbow_up=False -> solução "cotovelo para baixo"
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


# =========================
# Funções auxiliares de desenho
# =========================
def draw_arrow(ax, start, end, label=None, text_offset=(0.0, 0.0), lw=1.5):
    arrow = FancyArrowPatch(
        posA=start,
        posB=end,
        arrowstyle="-|>",
        mutation_scale=16,
        linewidth=lw,
        color="black",
    )
    ax.add_patch(arrow)
    if label is not None:
        mid = (np.array(start) + np.array(end)) / 2.0
        ax.text(mid[0] + text_offset[0], mid[1] + text_offset[1], label, fontsize=14)


def draw_joint(ax, center, r_outer=0.028, r_inner=0.010):
    outer = Circle(
        center, r_outer, facecolor="white", edgecolor="black", linewidth=1.8, zorder=5
    )
    inner = Circle(
        center, r_inner, facecolor="white", edgecolor="black", linewidth=1.8, zorder=6
    )
    ax.add_patch(outer)
    ax.add_patch(inner)


def draw_local_frame(ax, origin, angle, label_x, label_y, scale=0.12):
    """
    Desenha os eixos locais x_i e y_i.
    x local: ao longo do elo
    y local: perpendicular ao elo
    """
    ex = np.array([np.cos(angle), np.sin(angle)])
    ey = np.array([-np.sin(angle), np.cos(angle)])

    x_end = origin + scale * ex
    y_end = origin + scale * ey

    draw_arrow(ax, origin, x_end, label=label_x, text_offset=(0.01, -0.01), lw=1.0)
    draw_arrow(ax, origin, y_end, label=label_y, text_offset=(0.01, 0.01), lw=1.0)


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
    ax.text(label_pos[0], label_pos[1], label, fontsize=14)


def plot_arm(
    ax,
    theta1,
    theta2,
    target=None,
    title=None,
    show_local_frames=True,
    show_angles=True,
):
    O, P, Q = forward_kinematics(theta1, theta2)

    # Eixos globais
    draw_arrow(
        ax, (-0.02, 0.0), (1.05, 0.0), label=r"$X_0$", text_offset=(0.02, -0.05), lw=1.8
    )
    draw_arrow(
        ax, (0.0, -0.02), (0.0, 1.05), label=r"$Y_0$", text_offset=(-0.06, 0.02), lw=1.8
    )

    # Elos
    ax.plot([O[0], P[0]], [O[1], P[1]], color="black", linewidth=2.2)
    ax.plot([P[0], Q[0]], [P[1], Q[1]], color="black", linewidth=2.2)

    # Juntas
    draw_joint(ax, O)
    draw_joint(ax, P)

    # Tip simples como garra estilizada
    tip_dir = np.array([np.cos(theta1 + theta2), np.sin(theta1 + theta2)])
    tip_perp = np.array([-np.sin(theta1 + theta2), np.cos(theta1 + theta2)])
    wrist = Q - 0.02 * tip_dir
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

    # Labels dos pontos
    ax.text(O[0] - 0.015, O[1] - 0.065, "O", fontsize=14, fontweight="bold")
    ax.text(P[0] + 0.015, P[1] - 0.05, "P", fontsize=14, fontweight="bold")
    ax.text(Q[0] + 0.015, Q[1] - 0.01, "Q", fontsize=14, fontweight="bold")

    # Labels dos comprimentos
    mid1 = (O + P) / 2.0
    mid2 = (P + Q) / 2.0
    ax.text(mid1[0] - 0.02, mid1[1] + 0.04, r"$l_1$", fontsize=16)
    ax.text(mid2[0] - 0.01, mid2[1] + 0.04, r"$l_2$", fontsize=16)

    # Eixos locais
    if show_local_frames:
        draw_local_frame(ax, P, theta1, r"$x_1$", r"$y_1$", scale=0.12)
        draw_local_frame(ax, Q, theta1 + theta2, r"$x_2$", r"$y_2$", scale=0.12)

    # Ângulos
    if show_angles:
        theta1_deg = np.degrees(theta1)
        theta2_deg = np.degrees(theta2)

        # theta1: entre X0 e elo 1
        if abs(theta1_deg) > 1e-3:
            draw_angle_arc(
                ax=ax,
                center=O,
                radius=0.16,
                angle_start_deg=0,
                angle_end_deg=theta1_deg,
                label=r"$\theta_1$",
                label_pos=(0.12 * np.cos(theta1 / 2), 0.12 * np.sin(theta1 / 2) + 0.03),
            )

        # theta2: entre elo 1 e elo 2
        if abs(theta2_deg) > 1e-3:
            draw_angle_arc(
                ax=ax,
                center=P,
                radius=0.12,
                angle_start_deg=theta1_deg,
                angle_end_deg=theta1_deg + theta2_deg,
                label=r"$\theta_2$",
                label_pos=(
                    P[0] + 0.10 * np.cos(theta1 + theta2 / 2),
                    P[1] + 0.10 * np.sin(theta1 + theta2 / 2) + 0.03,
                ),
            )

    # Alvo
    if target is not None:
        ax.plot(target[0], target[1], marker="x", markersize=10, color="black", mew=2)
        ax.text(target[0] + 0.02, target[1] + 0.02, "target", fontsize=13)

    if title:
        ax.set_title(title, fontsize=14)

    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")


# =========================
# Geração das figuras
# =========================
def main():
    # Figura 1: posição inicial
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    plot_arm(
        ax1,
        theta1=THETA1_INIT,
        theta2=THETA2_INIT,
        target=None,
        title="Posição inicial",
    )
    fig1.tight_layout()
    fig1.savefig("arm_initial.png", dpi=300, bbox_inches="tight")
    fig1.savefig("arm_initial.svg", bbox_inches="tight")

    # Figura 2: alcançando alvo em (0.7, 0.6)
    theta1_goal, theta2_goal = inverse_kinematics_2link(
        TARGET[0], TARGET[1], elbow_up=True
    )

    fig2, ax2 = plt.subplots(figsize=(6, 6))
    plot_arm(
        ax2,
        theta1=theta1_goal,
        theta2=theta2_goal,
        target=TARGET,
        title="Braço alcançando o alvo em (0.7, 0.6)",
    )
    fig2.tight_layout()
    fig2.savefig("arm_target_0p7_0p6.png", dpi=300, bbox_inches="tight")
    fig2.savefig("arm_target_0p7_0p6.svg", bbox_inches="tight")

    # Mostrar na tela
    plt.show()

    print("Arquivos gerados:")
    print("- arm_initial.png")
    print("- arm_initial.svg")
    print("- arm_target_0p7_0p6.png")
    print("- arm_target_0p7_0p6.svg")
    print()
    print("Ângulos para o alvo:")
    print(f"theta1 = {theta1_goal:.6f} rad ({np.degrees(theta1_goal):.2f} deg)")
    print(f"theta2 = {theta2_goal:.6f} rad ({np.degrees(theta2_goal):.2f} deg)")


if __name__ == "__main__":
    main()
