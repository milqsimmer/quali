from two_link_arm_env import TwoLinkArmEnv


def sample_targets(seed, n=10):
    env = TwoLinkArmEnv(render=False, reward_mode="pure", seed=seed)
    targets = []
    for _ in range(n):
        env.reset()
        targets.append(tuple(env.target_pos))
    env.close()
    return targets


if __name__ == "__main__":
    t1 = sample_targets(0, n=8)
    t2 = sample_targets(0, n=8)
    t3 = sample_targets(1, n=8)

    print("seed=0 run1:", t1)
    print("seed=0 run2:", t2)
    print("seed=1 run :", t3)

    print("\nseed=0 run1 == run2 ?", t1 == t2)
    print("seed=0 run1 == seed=1 ?", t1 == t3)

    env = TwoLinkArmEnv(render=False, reward_mode="pure", seed=0)
    env.reset(seed=0)
    a = tuple(env.target_pos)

    env.reset(seed=0)
    b = tuple(env.target_pos)

    env.reset(seed=1)
    c = tuple(env.target_pos)
    print("a:", a)
    print("b (reseeded 0):", b)
    print("c (reseeded 1):", c)
