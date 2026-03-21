# Simulação de Braço Robótico 2D com PyBullet

Este projeto simula um braço robótico com dois graus de liberdade (2-DOF) utilizando a biblioteca PyBullet. O foco é a resolução da cinemática inversa (Inverse Kinematics) por meio de cálculos trigonométricos, com o objetivo de expandir para experimentos com Aprendizado por Reforço (Reinforcement Learning).

## 🔧 Estrutura do Robô

O braço robótico é composto por:
- Um link base fixo
- Dois segmentos (`link1` e `link2`) conectados por juntas rotacionais
- Um alvo (esfera vermelha) no espaço 3D
- A ponta do `link2` é considerada o end-effector (atuador final)

## 📐 Cinématica Inversa Analítica

Ao invés de usar o solver interno de IK do PyBullet, a solução dos ângulos é feita manualmente via trigonometria:

- Utiliza a posição do alvo no plano XY
- Calcula os ângulos `θ1` e `θ2` que levam a ponta do braço até o alvo
- Resolve com base nas fórmulas clássicas de cinemática de braços planarem 2-DOF

## 🚀 Próximos Passos

- Testes com múltiplas posições-alvo
- Criação de ambiente Gym customizado
- Treinamento com algoritmos de RL (PPO, DDPG, etc.)
- Geração de dados para imitation learning

## 📂 Estrutura
```
├── two_link_arm_env.py      # Ambiente Gym customizado para o braço 2-DOF
├── train_rl.py              # Script de treino PPO (modes: pure, pirl)
├── eval_rl.py               # Script de avaliação dos modelos treinados
├── scripts_test/
│   ├── demo_inicial.py      # Demo inicial em PyBullet (sem Gym/RL)
│   ├── test_env.py          # Teste de IK + TwoLinkArmEnv (distância ponta–alvo)
│   ├── test_tip_visual.py   # Debug visual da ponta/alvo com overlays
│   └── sanity_check.py      # Sanidade de Gym + NumPy
├── figs_env/                # Figuras do ambiente (layout, espaço de trabalho, etc.)
├── figs_resultados/         # Gráficos/resultados de avaliação
├── figs_treino/             # Curvas de treino (recompensa, etc.)
├── runs_pure/               # Saídas de treino PPO (modo "pure")
├── runs_pirl/               # Saídas de treino PPO (modo "pirl")
├── results/                 # CSVs/JSONs de avaliação gerados por eval_rl.py
├── two_link_arm.urdf        # Modelo URDF do braço
├── requirements.txt         # Dependências do projeto
├── requirements_legacy.txt  # Versão alternativa de dependências (legado)
├── AGENTS.md                # Guia para agentes de código
└── README.md                # Este arquivo
```

## ▶️ Como rodar

1. Instale as dependências (recomendado usar virtualenv):
   ```bash
   pip install -r requirements.txt
   ```

2. Simulação / demonstração inicial (PyBullet puro, sem Gym/RL):
   ```bash
   python scripts_test/demo_inicial.py
   ```

3. Testes rápidos e debug do ambiente:
   - Sanidade de Gym + NumPy:
     ```bash
     python scripts_test/sanity_check.py
     ```
   - Teste de IK + ambiente (vários alvos aleatórios):
     ```bash
     python scripts_test/test_env.py
     ```
   - Debug visual da ponta e do alvo (overlays na GUI do PyBullet):
     ```bash
     python scripts_test/test_tip_visual.py
     ```

## 🧠 Requisitos
- Python 3.8+
- Gym 0.21 (API legada: `reset() -> obs`, `step() -> (obs, reward, done, info)`)
- PyBullet
- NumPy
- stable_baselines3 (PPO)


## Treino e avaliação de RL

### Treino (PPO)

1. Modo "pure" (distância + penalidade leve de ação):
   ```bash
   python train_rl.py --mode pure --seed 0 --steps 300000
   ```

2. Modo "pirl" (distância + ação + penalidade em torque):
   ```bash
   python train_rl.py --mode pirl --seed 0 --steps 300000
   ```

3. Exemplos de sweep de seeds (para experimentos mais completos):
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

4. Saídas dos treinos:
   - `runs_pure/seed_<seed>/monitor.csv` e `runs_pure/seed_<seed>/ppo_model_<seed>.zip`.
   - `runs_pirl/seed_<seed>/monitor.csv` e `runs_pirl/seed_<seed>/ppo_model_<seed>.zip`.


### Avaliação

1. Avaliar um único modo:
   ```bash
   python eval_rl.py --mode pure --episodes 20 --print-episodes --render
   ```
   ou
   ```bash
   python eval_rl.py --mode pirl --episodes 100
   ```

2. Comparar diretamente os dois modos:
   ```bash
   python eval_rl.py --mode both --episodes 100 --out results/eval_results --seed 0
   ```

3. Exemplo de avaliação com foco em torque/“esforço”:
   ```bash
   python eval_rl.py --mode both --episodes 100 --out results_torque.csv
   ```

4. Saídas das avaliações:
   - `results/eval_results_<seed>.csv` (métricas por episódio).
   - `results/eval_results_<seed>_summary.json` (resumo agregado em JSON).


## Ajustes finos e dicas

- Se o braço mal se mexer durante o treino:
  - Aumente `force` no `POSITION_CONTROL` em `_apply_angles` dentro de `two_link_arm_env.py`,
    ou reduza a escala de ação (por exemplo, de ±0.1 para ±0.05).

- Se o movimento trepidar demais:
  - Reduza a escala de ação (ex.: ±0.05).
  - Aumente `max_episode_steps` no `TimeLimit` para 300 em `train_rl.py` (apenas para testes).

- Para “apimentar” o modo PIRL:
  - Adicione um termo extra de suavidade na recompensa, algo como
    `beta * (|Δθ1| + |Δθ2|)` por passo, se quiser penalizar variações bruscas de ângulo.
