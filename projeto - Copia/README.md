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
