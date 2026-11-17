"""
DQN simples para controle adaptativo de um cruzamento de 4 vias (v2).
Dependências: Python 3.10+ e NumPy.
Execução: python dqn_semaforos_v2.py

Novidades da v2:
- Comparação justa: cada ambiente tem seu próprio gerador aleatório
  (semente por episódio), então DQN e tempo fixo enfrentam exatamente
  as mesmas chegadas de veículos em cada episódio de avaliação.
- Avaliação com 100 episódios e diferença pareada (média, erro-padrão,
  número de vitórias).
- Dois cenários: tráfego balanceado e tráfego desbalanceado. Um agente
  é treinado para cada cenário.
"""
import random
import sys
from collections import deque
import numpy as np

# Evita caracteres quebrados no console do Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

SEED = 7

# Parâmetros de treino
EPISODES = 180
SECONDS_PER_EPISODE = 360
GAMMA = 0.95
LR = 0.001
BATCH_SIZE = 64
REPLAY_SIZE = 12000
TARGET_UPDATE = 10
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, 0.985

# Escala aplicada à recompensa apenas ao treinar
REWARD_SCALE = 0.1

# Avaliação
EVAL_EPISODES = 100
TRAIN_SEED_BASE = 1_000    # sementes dos ambientes de treino
EVAL_SEED_BASE = 100_000   # sementes dos ambientes de avaliação (separadas)

# Pesos da recompensa (ajustáveis nos experimentos)
ALPHA = 0.002   # espera acumulada
BETA = 0.035    # tamanho das filas
GAMMA_SWITCH = 0.50
DELTA = 1.00    # veículos atendidos
EPS_EMERGENCY = 3.0

# Cenários: probabilidades de chegada por segundo em N, S, L, O
SCENARIOS = {
    "Balanceado": [0.20, 0.20, 0.24, 0.24],
    "Desbalanceado": [0.40, 0.40, 0.10, 0.10],
}


class TrafficEnv:
    """Estado: filas[4], maior espera por via[4], fase, duração, emergência."""
    def __init__(self, seconds=360, arrival_rates=None):
        self.horizon = seconds
        # N, S, L, O: probabilidades de chegada por segundo
        self.arrival_rates = list(arrival_rates or SCENARIOS["Balanceado"])
        self.rng = random.Random(0)

    def reset(self, seed=None):
        # Gerador próprio: a sequência de chegadas depende só da semente,
        # não das decisões do agente nem do gerador global.
        self.rng = random.Random(seed)
        self.queues = [[] for _ in range(4)]  # cada veículo: [idade, emergência]
        self.phase = 0  # 0 = NS; 1 = LO
        self.phase_time = 0
        self.elapsed = 0
        self.total_served = 0
        self.total_reward = 0.0
        return self.state()

    def state(self):
        sizes = [len(q) for q in self.queues]
        oldest = [max([v[0] for v in q], default=0) for q in self.queues]
        emergency = [int(any(v[1] for v in q)) for q in self.queues]
        return np.array(
            [*[min(x, 40) / 40 for x in sizes],
             *[min(x, 120) / 120 for x in oldest],
             float(self.phase), min(self.phase_time, 60) / 60,
             *emergency], dtype=np.float32
        )

    def valid_actions(self):
        # 0 manter (+5s), 1 trocar (mínimo 10s), 2 prolongar (+10s)
        valid = [True, self.phase_time >= 10, self.phase_time <= 50]
        if self.phase_time >= 60:
            valid[0] = False
            valid[2] = False
        return np.array(valid, dtype=bool)

    def step(self, action):
        valid = self.valid_actions()
        if not valid[action]:
            action = 0 if valid[0] else 1
            invalid_penalty = 1.0
        else:
            invalid_penalty = 0.0

        switched = False
        if action == 1:
            # Amarelo de 3s, depois verde na direção oposta
            switched = True
            duration = 8  # 3s amarelo + 5s do novo verde
            self.phase = 1 - self.phase
            self.phase_time = 0
        elif action == 2:
            duration = 10
        else:
            duration = 5

        served = 0
        for t in range(duration):
            if self.elapsed >= self.horizon:
                break

            # Chegada estocástica de veículos (gerador do ambiente)
            for lane, rate in enumerate(self.arrival_rates):
                if self.rng.random() < rate:
                    is_emergency = self.rng.random() < 0.012
                    self.queues[lane].append([0, is_emergency])

            # Amarelo: nos 3 primeiros segundos da troca, ninguém é atendido
            yellow = switched and t < 3
            if not yellow:
                green_lanes = [0, 1] if self.phase == 0 else [2, 3]
                for lane in green_lanes:
                    if self.queues[lane]:
                        self.queues[lane].pop(0)
                        served += 1
                        self.total_served += 1

            # Atualiza idade de espera dos veículos ainda na fila
            for queue in self.queues:
                for vehicle in queue:
                    vehicle[0] += 1

            self.elapsed += 1
            self.phase_time += 1

        queue_total = sum(len(q) for q in self.queues)
        wait_sum = sum(v[0] for q in self.queues for v in q)
        emergency_late = sum(
            1 for q in self.queues for age, emergency in q
            if emergency and age > 30
        )
        reward = (
            -ALPHA * wait_sum
            - BETA * queue_total
            - GAMMA_SWITCH * float(switched)
            + DELTA * served
            - EPS_EMERGENCY * emergency_late
            - invalid_penalty
        )
        self.total_reward += reward
        done = self.elapsed >= self.horizon
        info = {"queue": queue_total, "wait_sum": wait_sum,
                "served": served, "switched": switched}
        return self.state(), float(reward), done, info


class ReplayBuffer:
    def __init__(self, capacity):
        self.data = deque(maxlen=capacity)

    def add(self, item):
        self.data.append(item)

    def sample(self, batch):
        return random.sample(self.data, batch)

    def __len__(self):
        return len(self.data)


class QNetwork:
    """MLP 14-64-64-3, ReLU e saída linear."""
    def __init__(self, input_dim=14, hidden=64, output_dim=3):
        self.W1 = np.random.randn(input_dim, hidden) * np.sqrt(2 / input_dim)
        self.b1 = np.zeros(hidden)
        self.W2 = np.random.randn(hidden, hidden) * np.sqrt(2 / hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = np.random.randn(hidden, output_dim) * np.sqrt(2 / hidden)
        self.b3 = np.zeros(output_dim)

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(0, z1)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, z2)
        q = h2 @ self.W3 + self.b3
        return q, (x, z1, h1, z2, h2)

    def train_batch(self, states, actions, targets, lr=LR):
        q, cache = self.forward(states)
        x, z1, h1, z2, h2 = cache
        grad = np.zeros_like(q)
        rows = np.arange(len(actions))
        grad[rows, actions] = 2 * (q[rows, actions] - targets) / len(actions)

        dW3 = h2.T @ grad
        db3 = grad.sum(axis=0)
        dh2 = grad @ self.W3.T
        dz2 = dh2 * (z2 > 0)
        dW2 = h1.T @ dz2
        db2 = dz2.sum(axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0)
        dW1 = x.T @ dz1
        db1 = dz1.sum(axis=0)

        # Gradient clipping simples
        grads = [dW1, db1, dW2, db2, dW3, db3]
        norm = np.sqrt(sum(np.sum(g * g) for g in grads))
        scale = min(1.0, 5.0 / (norm + 1e-8))
        for param, g in zip(
            [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3], grads
        ):
            param -= lr * scale * g

    def copy_from(self, other):
        for name in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(self, name, getattr(other, name).copy())


def choose_action(network, state, valid, epsilon):
    valid_ids = np.where(valid)[0].tolist()
    if epsilon > 0 and random.random() < epsilon:
        return int(random.choice(valid_ids))
    q_values, _ = network.forward(state.reshape(1, -1))
    q_values = q_values[0].copy()
    q_values[~valid] = -1e9
    return int(np.argmax(q_values))


def train(arrival_rates, episodes=EPISODES, verbose=True):
    # Sementes globais: inicialização da rede, exploração e replay
    random.seed(SEED)
    np.random.seed(SEED)

    env = TrafficEnv(SECONDS_PER_EPISODE, arrival_rates)
    online, target = QNetwork(), QNetwork()
    target.copy_from(online)
    memory = ReplayBuffer(REPLAY_SIZE)
    epsilon = EPS_START
    episode_rewards, episode_queues = [], []

    for episode in range(episodes):
        state = env.reset(seed=TRAIN_SEED_BASE + episode)
        done = False
        rewards, queues = [], []
        while not done:
            valid = env.valid_actions()
            action = choose_action(online, state, valid, epsilon)
            next_state, reward, done, info = env.step(action)
            next_valid = env.valid_actions()
            memory.add((state, action, reward * REWARD_SCALE,
                        next_state, done, next_valid))
            state = next_state
            rewards.append(reward)
            queues.append(info["queue"])

            if len(memory) >= BATCH_SIZE:
                batch = memory.sample(BATCH_SIZE)
                s, a, r, ns, d, masks = zip(*batch)
                s, ns = np.array(s), np.array(ns)
                a, r, d = np.array(a), np.array(r), np.array(d, dtype=bool)
                masks = np.array(masks, dtype=bool)
                q_next, _ = target.forward(ns)
                q_next = np.where(masks, q_next, -1e9)
                max_next = np.max(q_next, axis=1)
                y = r + GAMMA * max_next * (~d)
                online.train_batch(s, a, y)

        epsilon = max(EPS_END, epsilon * EPS_DECAY)
        if (episode + 1) % TARGET_UPDATE == 0:
            target.copy_from(online)
        episode_rewards.append(sum(rewards))
        episode_queues.append(float(np.mean(queues)))

        if verbose and (episode + 1) % 30 == 0:
            print(f"  Episódio {episode+1}/{episodes} | "
                  f"recompensa={episode_rewards[-1]:.1f} | epsilon={epsilon:.3f}")

    return online, episode_rewards, episode_queues


def evaluate(policy, arrival_rates, episodes=EVAL_EPISODES, fixed_time=False,
             seed_base=EVAL_SEED_BASE):
    """Retorna métricas por episódio (arrays). O episódio i usa a semente
    seed_base + i, igual para qualquer política, o que permite comparar
    de forma pareada."""
    out = {"reward": [], "queue": [], "served": [], "wait": [], "switches": []}
    for i in range(episodes):
        env = TrafficEnv(SECONDS_PER_EPISODE, arrival_rates)
        state = env.reset(seed=seed_base + i)
        done = False
        queue, served, wait, switches = [], 0, [], 0
        while not done:
            valid = env.valid_actions()
            if fixed_time:
                # Baseline: alterna NS/LO a cada 30s
                action = 1 if (env.phase_time >= 30 and valid[1]) else 0
            else:
                action = choose_action(policy, state, valid, 0.0)
            state, reward, done, info = env.step(action)
            queue.append(info["queue"])
            served += info["served"]
            wait.append(info["wait_sum"])
            switches += int(info["switched"])
        out["reward"].append(env.total_reward)
        out["queue"].append(np.mean(queue))
        out["served"].append(served)
        out["wait"].append(np.mean(wait))
        out["switches"].append(switches)
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def report(name, dqn, fixed):
    print(f"\n=== Cenário: {name} ({EVAL_EPISODES} episódios, mesmas sementes) ===")
    print(f"{'Métrica':<26}{'DQN':>12}{'Tempo fixo':>14}")
    rows = [
        ("Recompensa", "reward"),
        ("Fila média", "queue"),
        ("Espera acumulada média", "wait"),
        ("Atendidos/episódio", "served"),
        ("Trocas/episódio", "switches"),
    ]
    for label, key in rows:
        print(f"{label:<26}{dqn[key].mean():>12.2f}{fixed[key].mean():>14.2f}")

    # Diferença pareada na recompensa (DQN - fixo)
    diff = dqn["reward"] - fixed["reward"]
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    wins = int(np.sum(diff > 0))
    print(f"\nDiferença pareada de recompensa (DQN - fixo): "
          f"{diff.mean():+.2f} ± {1.96 * se:.2f} (IC95%)")
    print(f"DQN melhor em {wins}/{len(diff)} episódios")
    if abs(diff.mean()) > 1.96 * se:
        verdict = "DQN superior" if diff.mean() > 0 else "Tempo fixo superior"
        print(f"Conclusão: {verdict} (diferença estatisticamente distinguível de zero)")
    else:
        print("Conclusão: diferença não distinguível de zero (empate estatístico)")


if __name__ == "__main__":
    for name, rates in SCENARIOS.items():
        print(f"\n>>> Treinando cenário {name}: taxas N,S,L,O = {rates}")
        agent, rewards, queues = train(rates)
        dqn = evaluate(agent, rates)
        fixed = evaluate(None, rates, fixed_time=True)
        report(name, dqn, fixed)
