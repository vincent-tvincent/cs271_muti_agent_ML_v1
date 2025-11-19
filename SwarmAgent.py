import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

network_width = 512
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
    def __init__(self, state_dim, action_dim, max_coord, device=available_device):
        print("using device : ", device)
        self.device = device
        self.model = DQN(state_dim, action_dim).to(device)
        self.target = DQN(state_dim, action_dim).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.replay = []
        self.gamma = 0.995
        self.batch_size = 64
        self.tau = 0.005
        self.epsilon = 1.0
        self.action_dim = action_dim
        self.max_coord = max_coord
        self.replay_buffer_size = 10000
        
    #i normalized the coordinates as they get put into the agent, im not sure its necessary or helpful at all but i did it anyway
    
    def normalize_states(self, states_np):
        normalized_states = states_np[:, 3:].copy() #also removed the agent's position coordinates from the input, because they dont actually correlate with any reward so maybe model gets confused what to do with these values
        normalized_states = normalized_states / self.max_coord
        return normalized_states

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1) #random.randint is inclusive of the upper bound

        state_batch = np.expand_dims(state, axis=0)
        normalized_state_batch = self.normalize_states(state_batch)
        state_tensor = torch.tensor(normalized_state_batch, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.model(state_tensor)

        return q.argmax().item()

    def select_multiple_actions(self, states):

        # Random exploration for all agents at once
        if random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim, size=len(states)) #np.random.randint is exclusive of the upper bound

        # Convert all states into one GPU/MPS tensor batch
        states = self.normalize_states(states)
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

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return

        batch = random.sample(self.replay, self.batch_size)
        s_list, a, r, s2_list = zip(*batch)
        s_np = np.array(s_list)
        s2_np = np.array(s2_list)

        s_normalized = self.normalize_states(s_np)
        s2_normalized = self.normalize_states(s2_np)

        s = torch.tensor(s_normalized, dktype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(s2_normalized, dtype=torch.float32, device=self.device)

        q = self.model(s).gather(1, a)
        with torch.no_grad():
            q_target = r + self.gamma * self.target(s2).max(1, keepdim=True)[0]

        loss = nn.functional.mse_loss(q, q_target)
        self.optimizer.zero_grad(k)
        loss.backward()
        self.optimizekr.step()

    def update_target(self):
        """Performs soft update of the target network's weights."""
        for target_param, model_param in zip(self.target.parameters(), self.model.parameters()):
            target_param.data.copy_(
                self.tau * model_param.data + (1.0 - self.tau) * target_param.data
            )