# Relatorio: Ambiente TwoLinkArm, PPO e PI-Rewards

Este relatorio descreve em nivel conceitual e pratico como os principais arquivos do projeto
se conectam e qual a teoria de reinforcement learning (RL) e PI-rewards envolvida.

- `two_link_arm.urdf`: modelo fisico do braco de 2 graus de liberdade (2-DOF) no PyBullet.
- `two_link_arm_env.py`: ambiente Gym que encapsula a simulacao fisica e define a tarefa de RL.
- `train_rl.py`: script de treino com PPO, usando o ambiente.
- `eval_rl.py`: script de avaliacao e comparacao entre modos de recompensa `pure` e `pirl`.

---

## 1) Robo fisico: `two_link_arm.urdf`

O arquivo URDF e um XML que descreve a geometria, inercia e juntas do robo simulado.

- Links principais:
  - `base_link`: base fixa com um cubo pequeno apenas para visualizacao.
  - `link1` e `link2`: segmentos de 0.5 m de comprimento (ver `box size="0.5 ..."` e o `origin` no meio).
- Massas e inercia:
  - Massas simples (`0.2` e `0.15`) e inercia `0.01` para cada link (modelo simplificado).
- Juntas:
  - `joint1`: junta revoluta entre `base_link` e `link1`, eixo `z` (movimento planar no plano XY), limites `[-pi, pi]`.
  - `joint2`: junta revoluta entre `link1` e `link2`, tambem em torno do eixo `z`.

Em termos de cinematica, o robo e um braco planar classico com dois elos de 0.5 m, girando no
plano XY. O URDF nao contem nada de RL: ele apenas define o corpo que o PyBullet vai simular.

---

## 2) Ambiente Gym: `TwoLinkArmEnv` (`two_link_arm_env.py`)

`TwoLinkArmEnv` herda de `gym.Env` e integra PyBullet, o URDF e a definicao de estados, acoes e
recompensas para RL.

### 2.1 Inicializacao

- Conexao e simulacao:
  - Conecta ao PyBullet (`p.GUI` ou `p.DIRECT`).
  - Carrega `plane.urdf` (chao) e `two_link_arm.urdf` na origem.
  - Define gravidade `-9.8 m/s^2` no eixo z.
- Parametros geometricos:
  - `l1 = l2 = 0.5` (comprimentos dos elos).
  - `offset_local = [l2, 0, 0]` para chegar na ponta do ultimo link.
  - `success_threshold = 0.05` (5 cm) para considerar que o alvo foi atingido.
- Espaco de observacao Gym:
  - Vetor: `[theta1, theta2, target_x, target_y]`.
  - `Box([-pi, -pi, -1.5, -1.5], [pi, pi, 1.5, 1.5])`.
- Espaco de acao:
  - Vetor de incrementos de angulo: `acao = [delta_theta1, delta_theta2]`.
  - `Box([-0.1, -0.1], [0.1, 0.1])` (radianos por passo).

### 2.2 `reset()`

- Zera os angulos das juntas: `theta1 = theta2 = 0`.
- Sorteia um alvo `_sample_target()`:
  - `x ~ U(0.1, 0.9)` e `y ~ U(-0.5, 0.5)`.
  - Garante que `sqrt(x^2 + y^2) <= l1 + l2` (alvo dentro do alcance maximo).
- Aplica os angulos com `_apply_angles` (controle de posicao nas duas juntas) e roda
  `p.stepSimulation()`.
- Cria uma esfera vermelha no alvo com `_create_target_visual()`.
- Retorna a observacao inicial `[theta1, theta2, target_x, target_y]`.

### 2.3 `step(action)`

1. Atualiza angulos internos:
   - `theta1 <- clip(theta1 + action[0], -pi, pi)`.
   - `theta2 <- clip(theta2 + action[1], -pi, pi)`.
2. Aplica os angulos no PyBullet com `POSITION_CONTROL` e `force=20` para cada junta.
3. Roda um passo de simulacao (`p.stepSimulation()`).
4. Calcula a posicao da ponta:
   - `_get_end_effector_pos()` usa `p.getLinkState` do `link2` e `offset_local` para achar a
     ponta real.
   - Considera apenas `(x, y)` da ponta para a distancia.
5. Calcula distancia ao alvo:
   - `dist_t = || (x_ee, y_ee) - (target_x, target_y) ||_2`.
6. Lembra os torques e velocidades instantaneos:
   - `js = p.getJointStates(self.robot, [0, 1])`.
   - Para cada junta, `appliedMotorTorque` e o quarto elemento.
   - `tau1 = |js[0][3]|`, `tau2 = |js[1][3]|`.
   - `tau_sum = |tau1| + |tau2| = ||tau_t||_1` (norma L1 do torque no passo t).
   - Velocidades angulares: `omega1 = js[0][1]`, `omega2 = js[1][1]`.
   - Potencia instantanea aproximada: `power_t = |tau1 * omega1| + |tau2 * omega2|`.
7. Recompensa (veja Secao 5 para teoria):
   - Define `lambda_a` (argumento do construtor; padrao `0.001`) como o peso da penalidade de
     acao, norma 2 do vetor de incrementos.
   - Modo `"pure"`:
     - `reward_t = -dist_t - lambda_a * ||action_t||_2`.
   - Modo `"pirl"`:
     - Define `alpha_tau` (argumento do construtor; padrao `0.0005`) como peso da penalidade
       de torque.
     - `reward_t = -dist_t - lambda_a * ||action_t||_2 - alpha_tau * tau_sum`.
8. Termino do episodio:
   - `done = (dist_t < success_threshold)`.
   - Nao ha limite de passos aqui; o limite de 200 passos e imposto por `TimeLimit` nos
     scripts de treino e avaliacao.
9. Dicionario `info` retornado:
   - `distance`, `final_distance` (iguais no passo atual), `is_success` (0 ou 1),
     `tau1`, `tau2`, `tau_sum`, `omega1`, `omega2`, `power`.

Em resumo, `TwoLinkArmEnv` define um MDP continuo com:

- Estados `s_t = (theta1_t, theta2_t, target_x, target_y)`.
- Acoes `a_t = (delta_theta1_t, delta_theta2_t)`.
- Dinamica dada pela fisica do PyBullet.
- Recompensa `r_t` que muda conforme o modo (`pure` ou `pirl`).

---

## 3) Treino de RL com PPO: `train_rl.py`

Este script treina um agente PPO no ambiente `TwoLinkArmEnv`, usando um dos dois modos de
recompensa.

### 3.1 Linha de comando e configuracao

- Argumentos principais:
  - `--mode`: `"pure"` ou `"pirl"` (define a funcao de recompensa do ambiente).
  - `--steps`: numero total de passos de treino (padrao `300_000`).
  - `--seed`: semente para reproducibilidade.
  - `--lambda-a`: peso da penalidade de acao `lambda_a` (padrao `0.001`).
  - `--alpha-tau`: peso da penalidade de torque `alpha_tau` (padrao `0.0005`, so usado no
    modo `"pirl"`).
  - `--run-tag`: sufixo opcional para distinguir execucoes com a mesma seed (gera pastas
    `seed_<seed>_<run-tag>`).
- Reproducibilidade:
  - `set_global_seed(seed)` fixa `random` e `numpy`.

### 3.2 Diretorios e logs

- Diretorio da execucao:
  - Sem `run-tag`: `runs_<mode>/seed_<seed>`.
  - Com `run-tag`: `runs_<mode>/seed_<seed>_<run-tag>`.
- Arquivos gerados:
  - `monitor.csv`: log por episodio (via `Monitor`).
  - `ppo_model_<seed>.zip`: modelo PPO treinado (weights e parametros).

### 3.3 Ambiente e wrappers

- Ambiente base:
  - `env = TwoLinkArmEnv(render=False, reward_mode=args.mode, lambda_a=args.lambda_a,
    alpha_tau=args.alpha_tau)`.
- Limite de passos por episodio:
  - `env = TimeLimit(env, max_episode_steps=200)`.
- Monitoramento de episodios:
  - `env = Monitor(env, filename=monitor_path, info_keywords=("is_success", "final_distance"))`.
  - O wrapper acumula as informacoes de `info` no final de cada episodio e grava no CSV.

### 3.4 PPO e otimizacao

- Modelo PPO:
  - `PPO("MlpPolicy", env, ...)` com hiperparametros classicos (`learning_rate=3e-4`,
    `n_steps=2048`, `batch_size=256`, etc.).
  - A rede recebe o vetor de observacao (4 dimensoes) e produz uma distribuicao gaussiana
    sobre as acoes (2 dimensoes).
- Objetivo de PPO (visao de alto nivel):
  - Maximizar o retorno esperado `E[sum gamma^t r_t]` ajustando a politica `pi_theta(a|s)`.
  - Usa uma perda "clipped" para nao mudar demais a politica entre iteracoes.
- Treino:
  - `model.learn(total_timesteps=args.steps)` executa interacoes com o ambiente e atualiza
    a politica.
  - Depois, o modelo e salvo em `model_path`.

Importante: o algoritmo PPO e o mesmo em ambos os modos. A unica mudanca entre `pure` e `pirl`
e a definicao de `r_t` no ambiente.

---

## 4) Avaliacao e comparacao: `eval_rl.py`

`eval_rl.py` carrega modelos PPO treinados (pure ou pirl), roda varios episodios e gera
estatisticas detalhadas, inclusive sobre o esforco em torque e a energia aproximada por
episodio.

### 4.1 Funcao `evaluate`

- Carregamento do modelo:
  - `model_path` e construido com `get_model_path(mode, seed, run_tag)` e aponta para:
    - `runs_<mode>/seed_<seed>/ppo_model_<seed>.zip` (sem tag) ou
    - `runs_<mode>/seed_<seed>_<run-tag>/ppo_model_<seed>.zip` (com tag).
  - Cria `env = TwoLinkArmEnv(render=render, reward_mode=mode)`.
  - Aplica `TimeLimit(env, max_episode_steps=max_steps)`.
  - `model = PPO.load(model_path, env=env)`.

- Loop de episodios:
  - Para cada episodio:
    - `obs = env.reset()`.
    - Enquanto nao terminar:
      - `action, _ = model.predict(obs, deterministic=True)`.
      - `obs, reward, done, info = env.step(action)`.
      - Acumula retorno `ep_ret` e comprimento `ep_len`.
      - Soma de esforcos de torque: `ep_tau_sum += tau_sum` (lido de `info`).
      - Energia: `ep_energy += power_t * dt` (usando `power` vindo do `info`).
  - Ao final:
    - `final_dist = last_info["distance"]`.
    - `success = 1` se `final_dist < 0.05`, senao `0`.
    - Esforco medio de torque no episodio:
      - `E_i = ep_tau_sum / ep_len` (se `ep_len > 0`).

- Estatisticas agregadas (por modo):
  - `success_rate` (taxa de sucesso).
  - `mean_final_dist` e `std_final_dist`.
  - `mean_ep_len` e `std_ep_len`.
  - `mean_tau_effort`, `std_tau_effort`, `median_tau_effort` (metricas do esforco medio de torque).
  - `mean_energy` e `std_energy` (energia aproximada por episodio).

### 4.2 Salvando resultados

- `save_csv(csv_path, rows, summaries)`:
  - Escreve um CSV por episodio:
    - colunas: `mode`, `seed`, `episode`, `success`, `final_distance`, `return`, `length`,
      `mean_tau_sum`, `tau_sum_total`, `energy`.
  - Escreve um JSON de resumo ao lado do CSV (mesmo prefixo, sufixo `_summary.json`).

- `print_summary_table(summaries)`:
  - Imprime uma tabela alinhada com colunas como
    `mode`, `seed`, `episodes`, `success_rate`, `mean_final_dist`, `mean_ep_len`,
    `mean_tau_effort`, `mean_energy`.

### 4.3 Execucao via linha de comando

- Argumentos principais:
  - `--mode`: `"pure"`, `"pirl"` ou `"both"`.
  - `--seed`: qual modelo carregar (compatível com o treino).
  - `--episodes`: episodios por modo.
  - `--render`: se verdadeiro, mostra a simulacao.
  - `--max-steps`: limite de passos (TimeLimit).
  - `--out`: prefixo do CSV de saida.
  - `--print-episodes`: imprime metrica por episodio.
  - `--run-tag`: sufixo opcional para casar com o `--run-tag` usado no treino.

Se `mode="both"`, o script avalia sequencialmente os dois modelos (`pure` e `pirl`) e junta
os resultados no mesmo CSV e JSON, facilitando a comparacao.

---

## 5) Teoria: RL, torque e PI-Rewards

### 5.1 MDP e PPO

O problema e formulado como um MDP continuo:

- Espaco de estados `S`:
  - `s = (theta1, theta2, target_x, target_y)`.
- Espaco de acoes `A`:
  - `a = (delta_theta1, delta_theta2)`.
- Dinamica `P(s_{t+1} | s_t, a_t)`:
  - Dada pela fisica do PyBullet com controle de posicao nas juntas.
- Recompensa `R(s_t, a_t)` ou `r_t`:
  - Definida conforme o modo: `pure` ou `pirl`.
- Fator de desconto `gamma` (no PPO, tipicamente `0.99`).

O objetivo de PPO e encontrar uma politica parametrizada `pi_theta(a|s)` que maximize o
retorno esperado:

```text
J(theta) = E_pi [ sum_{t=0}^infinito gamma^t r_t ].
```

PPO usa um estimador de gradiente de politica com uma funcao de valor (critic) e uma
perda "clipped" para estabilizar as atualizacoes, mas o ponto central aqui e que ele depende
apenas das observacoes e das recompensas que o ambiente devolve.

### 5.2 Recompensa pura (modo `pure`)

No modo puro, a recompensa por passo e:

```text
r_t^pure = - dist_t - lambda_a * ||action_t||_2,
```

onde:

- `dist_t` e a distancia euclidiana entre a ponta do braco e o alvo.
- `||action_t||_2` e o tamanho do incremento de angulo aplicado.
- `lambda_a = 0.001` e um fator pequeno que desencoraja acoes muito grandes.

Interpretacao:

- O agente e incentivado a minimizar a distancia ao alvo o mais rapido possivel.
- Ha uma penalidade leve para nao explodir as acoes, mas o esforco fisico real (torque)
  nao entra explicitamente na funcao de retorno.

### 5.3 Recompensa com torque (modo `pirl`)

No modo `pirl`, a recompensa se torna:

```text
r_t^pirl = - dist_t - lambda_a * ||action_t||_2 - alpha_tau * ||tau_t||_1,
```

onde:

- `||tau_t||_1 = |tau1_t| + |tau2_t|` e a norma L1 dos torques aplicados nas duas juntas.
- `alpha_tau = 0.0005` e o peso de penalidade de torque.

Interpretacao:

- Continua havendo incentivo para aproximar a ponta do alvo.
- Alem disso, o agente paga um custo adicional toda vez que produz torques grandes.
- Isso introduz um trade-off entre:
  - sucesso na tarefa (chegar perto do alvo), e
  - esforco fisico (magnitude do torque ao longo da trajetoria).

Do ponto de vista de RL, `alpha_tau` controla o quanto o algoritmo se preocupa com conforto/
esforco vs. desempenho bruto na tarefa.

### 5.4 PI-Rewards e comparacao entre politicas

A ideia de PI-Rewards (ou PIRL) e usar informacao fisica (como torque, energia, etc.) como
parte da funcao de recompensa ou como metrica adicional na avaliacao.

Neste projeto:

- Durante o treino:
  - Modo `pure`: a funcao de recompensa nao usa torque, mas o ambiente ainda registra `tau_sum`.
  - Modo `pirl`: o torque entra diretamente na recompensa como custo.
- Durante a avaliacao (`eval_rl.py`):
  - Tanto modelos `pure` quanto `pirl` sao avaliados com as mesmas metricas:
    - `success_rate`, `mean_final_dist`.
    - `mean_tau_effort`, `tau_sum_total` e derivados.

Assim, e possivel comparar:

- Quanta recompensa cada politica obtem (no seu proprio modo).
- Quanta energia/esforco de torque cada uma consome em media.

Em geral, espera-se que:

- Policas treinadas com `pure` atinjam o alvo de forma rapida ou agressiva, mas com maior
  esforco medio de torque.
- Policas treinadas com `pirl` possam aceitar caminhos um pouco mais longos ou retornos
  puros ligeiramente menores, em troca de trajetorias mais suaves e menor esforco em torque.

Esse e o nucleo da trade-off que o PI-Rewards busca explicitar: otimizar nao apenas o sucesso
da tarefa, mas tambem criterios fisicos relevantes para robotica (torque, energia, conforto).

---

### 5.5 Decisoes de desenho do experimento

Algumas decisoes especificas foram tomadas para deixar o experimento conceitualmente mais
limpo e as comparacoes entre modos mais honestas:

- Separar objetivos de treino e metricas de avaliacao:
  - `mean_return` **nao** e usado para comparar `pure` vs `pirl`, pois cada modo tem uma
    funcao de recompensa diferente (o termo de torque aparece so em `pirl`).
  - Em vez disso, a comparacao entre modos e feita com base em `success_rate`,
    `mean_final_dist`, `mean_tau_effort` e `mean_energy`, que sao interpretaveis em termos
    fisicos e identicos para ambos.

- Medir energia de forma fisicamente coerente:
  - A energia aproximada por episodio e definida como a integral discreta da potencia: soma de
    `power_t * dt`, onde `power_t = |tau1_t * omega1_t| + |tau2_t * omega2_t|`.
  - O passo de tempo `dt` nao e fixado "no chute": e lido de
    `pybullet.getPhysicsEngineParameters()["fixedTimeStep"]`, garantindo que a escala de
    energia respeita o configurado no motor fisico.

- Parametrizar e explicitar os pesos de recompensa:
  - Os pesos `lambda_a` (acao) e `alpha_tau` (torque) sao argumentos explicitos do
    `TwoLinkArmEnv` e dos scripts de treino (`--lambda-a`, `--alpha-tau`).
  - Isso facilita explorar diferentes configuracoes de PI-reward e registrar exatamente
    quais valores foram usados em cada execucao (via `--run-tag` e a estrutura de pastas).

Essas escolhas visam separar claramente o que e parte do objetivo de treino (recompensa) do que
e parte da avaliacao final (metricas fisicas) e tornar os resultados mais comparaveis e
reprodutiveis.
