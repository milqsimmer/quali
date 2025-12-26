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
├── main.py # Código principal da simulação 
├── two_joint_robot.urdf # Modelo URDF do braço 
├── README.md # Este arquivo 
├── requirements.txt # Dependências do projeto
```

## ▶️ Como rodar

1. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute a simulação:

```python
  python main.py
```

## 🧠 Requisitos

- Python 3.8+
- PyBullet
- NumPy

## Novo

1. rodar treinos e results:

> treino

```python
  python train_rl.py --mode pure
  python train_rl.py --mode pirl
```

> eval

```python
  python eval_rl.py --mode pure --episodes 20 --print-episodes --render
```

  ou

```python
  python eval_rl.py --mode both --episodes 100
```

2. Se o braço mal se mexer: aumente force no POSITION_CONTROL (em _apply_angles) ou reduza a escala de ação (±0.05).
3. Se trepidar: reduza a escala de ação (ex.: ±0.05) e aumente max_episode_steps para 300 (apenas teste).
4. Se quiser “apimentar” o PIRL sem esforço: adicione um termo de suavidade (ex.: beta * (|Δθ1| + |Δθ2|)).

## Avaliação v2.0 (pré-qualificação):

```python
  python eval_rl.py --mode both --episodes 100 --out results_torque.csv
```

# Versão 3.0 (pós-qualificação)

## Testes pós melhorias

### Teste de amostras do alvo

=== Teste A2: sampler de alvo ===
Env: TwoLinkArmEnv (from two_link_arm_env)
n=20000, seed=0
l1=0.5000, l2=0.5000
anel: r_min=0.0200, r_max=0.9800
min_dist_tip0=0.0700
ik_check=OFF

Violação anel: 0/20000
Violação min_dist: 0/20000

r (min/mean/max) = 0.0204 / 0.6546 / 0.9799
d_tip0 (min/mean/max) = 0.0712 / 0.7553 / 1.3884


## Rodando o Treino
```python
  python train_rl.py --mode pirl --seed 0
```

ou, para 5 seeds:

```python
  python train_many_seeds.py --mode pure --steps 300000 --seeds 0,1,2,3,4
  python train_many_seeds.py --mode pirl --steps 300000 --seeds 0,1,2,3,4
```

## Avaliação v3.0 (pós-qualificação):

```python
  python eval_rl.py --mode both --train-seed 0 --eval-seed-base 1000 --episodes 200 --out results/eval_train0.csv

```

ou com executavel > No PowerShell, dentro da pasta do projeto:
```
.\run_eval.ps1
```

## Coletar todos os resultados
Depois de rodar vários evals (ex.: results/eval_seed0_train0.csv, results/eval_seed0_train1.csv, ...):
```
python merge_results.py --pattern "results/*.csv" --out-dir results_agg
```

# Versão 4.0 - torque control

## Treino
```
.\run_train_grid.ps1
# ou
.\run_train_grid.ps1 -Steps 500000
```
## Eval
```
.\run_eval_grid.ps1

# Com episodes maior:
.\run_eval_grid.ps1 -Episodes 500
```

## Resultados
```
python aggregate_results.py --results-dir results --out-dir results

# Se quiser gerar tabela LaTeX:
python aggregate_results.py --results-dir results --out-dir results --latex results/table_across_seeds.tex
```

### Como usar make_latex_tables_d0bins.py:

1) Um único .tex com todos os bins:

python make_latex_tables_d0bins.py --in-csv results/aggregated_d0bin_across_seeds.csv --out-tex results/table_d0bins.tex


2) Um .tex por bin (recomendado se quiser colocar várias tabelas menores):

python make_latex_tables_d0bins.py --in-csv results/aggregated_d0bin_across_seeds.csv --out-dir results/tables_d0 --split


### Testar os 4 modos:
smoke test 1:
> python train_rl.py --exp-id ctrl-direct__pi-0_test --control direct --seed 0 --steps 20000

> python train_rl.py --exp-id ctrl-direct__pi-1_test --control direct --use-pi-reward --alpha-pi 0.0005 --pi-metric tau_l1 --seed 0 --steps 20000

> python eval_rl.py --exp-id ctrl-direct__pi-0_test --train-seed 0 --eval-seed-base 1000 --episodes 50 --out-dir results_smoke

> python eval_rl.py --exp-id ctrl-direct__pi-1_test --train-seed 0 --eval-seed-base 1000 --episodes 50 --out-dir results_smoke

smoke test 2:
  - smoke_validate_before_big_train.py
   > python smoke_validate_before_big_train.py --seed 0 --steps 50




