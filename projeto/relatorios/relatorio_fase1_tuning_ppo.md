# Relatório Fase 1 – Tuning de PPO para o braço de 2 elos

## Contexto

Os resultados de avaliação atuais (arquivos em `tabelas/` e `results/`) mostram:

- Taxa de sucesso média em torno de 10% tanto para `pure` quanto para `pirl`.
- Distância final média da ponta ao alvo em torno de 0,30 m, bem acima do limiar de sucesso (`success_threshold = 0.05`).
- Forte variação entre seeds (por exemplo, `success_rate` indo de ~0.055 até ~0.195 para o modo `pure`).

Esses sinais indicam que, com o orçamento atual de treino (`--steps 300_000`) e os hiperparâmetros padrão de PPO, o agente não está conseguindo aprender uma política robusta que alcance o alvo de forma consistente.

Este relatório documenta a **Fase 1** do plano de melhorias: ajustes nos hiperparâmetros de PPO, mantendo a estrutura do ambiente e da recompensa.

## Objetivo desta fase

- Aumentar a estabilidade e a eficiência do treinamento PPO.
- Melhorar a taxa de sucesso já durante o treinamento, sem ainda alterar a função de recompensa nem o critério de sucesso.
- Facilitar futuros experimentos adicionando hiperparâmetros configuráveis via linha de comando.

## O que foi alterado

Arquivo principal modificado: `train_rl.py`.

1. **Exposição de hiperparâmetros de PPO via CLI**

Foram adicionados novos argumentos ao parser de linha de comando:

- `--learning-rate` (`dest=learning_rate`, `default=2e-4`)
- `--n-steps` (`dest=n_steps`, `default=2048`)
- `--batch-size` (`dest=batch_size`, `default=128`)
- `--n-epochs` (`dest=n_epochs`, `default=20`)
- `--gamma` (`default=0.99`)
- `--gae-lambda` (`dest=gae_lambda`, `default=0.95`)
- `--clip-range` (`dest=clip_range`, `default=0.2`)

Esses argumentos permitem controlar diretamente os principais hiperparâmetros do PPO sem alterar o código.

2. **Novo conjunto de hiperparâmetros padrão (Fase 1)**

Na criação do modelo PPO:

```python
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=args.seed,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
)
```

Os valores padrão desses argumentos (quando não especificados na CLI) foram ajustados para:

- `learning_rate = 2e-4` (antes: `3e-4`)
- `n_steps = 2048` (mantido em relação ao código anterior)
- `batch_size = 128` (antes: `256`)
- `n_epochs = 20` (antes: `10`)
- `gamma = 0.99` (mantido)
- `gae_lambda = 0.95` (mantido)
- `clip_range = 0.2` (mantido)

## Justificativa das alterações

### 1. Taxa de aprendizado (`learning_rate`)

- Valores em torno de `3e-4` são comuns em exemplos da própria Stable-Baselines3, mas podem ser relativamente agressivos em tarefas sensíveis, levando a políticas instáveis ou que não refinam bem o comportamento.
- Reduzir para `2e-4` tende a tornar o treinamento um pouco mais estável, permitindo que o agente refine melhor a política ao longo de mais passos.

### 2. Tamanho do batch (`batch_size`)

- Com `n_steps = 2048` e `batch_size = 256`, cada rollout produz 2048 amostras e o otimizador faz apenas 8 atualizações por época.
- Reduzindo o `batch_size` para `128`, aumentamos o número de updates por época para 16, explorando melhor os dados coletados em cada rollout e potencialmente melhorando a amostragem do gradiente.

### 3. Número de épocas de otimização (`n_epochs`)

- Com `n_epochs = 10`, cada amostra é vista em média 10 vezes por atualização de parametros.
- Aumentar para `20` torna o treinamento mais "data-efficient" por rollout, o que é útil quando o orçamento de passos totais (`--steps`) é relativamente limitado.
- A literatura de PPO e a implementação da Stable-Baselines3 sugerem valores entre 3 e 20; estamos escolhendo o extremo mais alto visando maximizar o aproveitamento dos dados coletados.

### 4. Manutenção de `gamma`, `gae_lambda` e `clip_range`

- Os valores anteriores (`gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`) já são amplamente usados como padrão em tarefas contínuas e foram mantidos nesta fase.
- Modificações nesses parâmetros serão consideradas em fases posteriores, se necessário, para incentivar ainda mais sucessos rápidos e estabilidade.

## Referências utilizadas

- Stable-Baselines3 – PPO documentation e exemplos oficiais:
  - https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
  - https://github.com/DLR-RM/stable-baselines3
- Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017.
- Prática comum em benchmarks contínuos (MuJoCo/Reacher-like) usando PPO, que tipicamente utilizam:
  - `gamma` em torno de 0.99,
  - `gae_lambda` entre 0.9 e 0.95,
  - `clip_range` em torno de 0.1–0.2,
  - `learning_rate` na faixa 1e-4–3e-4,
  - `n_steps` na faixa 1024–4096.

## Como rodar os experimentos desta fase

Sugestão de configuração para a **Fase 1** (mantendo os novos padrões):

```bash
# Exemplo para modo pure, seed 0
python train_rl.py --mode pure --seed 0 --steps 1000000

# Exemplo para modo pirl, seed 0
python train_rl.py --mode pirl --seed 0 --steps 1000000

# Repetir para seeds 1..4 para cada modo
```

Opcionalmente, é possível sobrescrever algum hiperparâmetro, por exemplo:

```bash
python train_rl.py --mode pure --seed 0 --steps 1000000 \
    --learning-rate 3e-4 --batch-size 256
```

Após o treino, usar os scripts existentes (`eval_rl.py`, análises em `tabelas/`) para recomputar as métricas de sucesso e distância final.

## Resultados desta fase

> **A ser preenchido após rodar os experimentos com a nova configuração.**

Sugestão de itens a registrar aqui assim que os treinos forem concluídos:

- Tabela com `success_rate_mean`, `final_distance_mean` e `return_mean` antes vs. depois para cada modo (`pure` e `pirl`).
- Gráficos ou estatísticas mostrando a evolução da taxa de sucesso ao longo do treino (a partir de `monitor.csv`).
- Comentário qualitativo sobre estabilidade entre seeds (por exemplo: variação máxima da taxa de sucesso entre seeds).
