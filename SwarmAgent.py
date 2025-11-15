import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

network_width = 516
# ------------------------------
# 2. DQN Model
# ------------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, network_width),
            nn.ReLU(),
            nn.Linear(network_width, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ------------------------------
# 3. Shared DQN Agent
# ------------------------------
available_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

class Agent:
    def __init__(self, state_dim, action_dim, device=available_device):
        print("using device : ", device)
        self.device = device
        self.model = DQN(state_dim, action_dim).to(device)
        self.target = DQN(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.replay = []
        self.gamma = 0.95
        self.batch_size = 64
        self.epsilon = 1.0

        self.replay_buffer_size = 100000

        if self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler()

    def epsilon_decay(self, epsilon_min, epsilon_decay):
        self.epsilon = max(epsilon_min, self.epsilon * epsilon_decay)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 6)

        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.model(state)

        return q.argmax().item()

    def select_multiple_actions(self, states):

        # Random exploration for all agents at once
        if random.random() < self.epsilon:
            return np.random.randint(0, 7, size=len(states))

        # Convert all states into one GPU/MPS tensor batch
        states = torch.tensor(states, dtype=torch.float32, device=self.device)

        # Forward pass through model (batch inference)
        with torch.no_grad():
            q_values = self.model(states)

        # Pick best action for each agent
        actions = q_values.argmax(dim=1).cpu().numpy()
        return actions

    def store(self, s, a, r, s2):
        self.replay.append((s, a, r, s2))
        if len(self.replay) > self.replay_buffer_size:
            self.replay.pop(0)

    def train_step(self, done=False):
        if len(self.replay) < self.batch_size and not done:
            return
        batch = random.sample(self.replay, self.batch_size)
        s, a, r, s2 = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)

        if self.device.type == "cuda":
            with torch.no_grad():
                q_target = r + self.gamma * self.target(s2).max(1, keepdim=True)[0]

            with torch.cuda.amp.autocast():
                q = self.model(s).gather(1, a)
                loss = nn.functional.mse_loss(q, q_target)

            self.optimizer.zero_grad()

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)

            self.scaler.update()

        else:
            q = self.model(s).gather(1, a)
            with torch.no_grad():
                q_target = r + self.gamma * self.target(s2).max(1, keepdim=True)[0]

            loss = nn.functional.mse_loss(q, q_target)

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())