import numpy as np

# ------------------------------
# 1. Simple 3D Swarm Environment
# ------------------------------
class SwarmEnv:
    def __init__(self, n_agents=5, space_size=10):
        self.n_agents = n_agents
        self.space_size = space_size
        self.goal_reward = 1.0
        self.collision_reward = -1.0
        self.distance_reward_factor = 2.0

        self.action_set = np.array([
            [0, 0, 0],
            [0.5, 0, 0], [-0.5, 0, 0],
            [0, 0.5, 0], [0, -0.5, 0],
            [0, 0, 0.5], [0, 0, -0.5]
        ])
        self.goal = np.zeros([self.n_agents, 3])
        self.done = np.zeros(self.n_agents)
        self.positions = np.zeros([self.n_agents, 3])
        self._observing_environment()
        self.reset()

    def set_random_goals(self):
        self.goal = np.random.randint(0, self.space_size, size=self.goal.shape)

    def reset(self):
        self.done = np.zeros(self.n_agents)
        self.positions = np.random.randint(0, self.space_size // 2, (self.n_agents, 3))
        return self._observing_environment()

    def _observing_environment(self):
        # Each agent observes its position + relative goal + nearest neighbor
        observations = []
        for agent_id in range(self.n_agents):
            current_position = self.positions[agent_id]
            relative_goal = self.goal[agent_id] - current_position
            distances = np.linalg.norm(self.positions - current_position, axis=1)
            nearest = self.positions[np.argsort(distances)[1]] - current_position  # skip itself
            observations.append(np.concatenate([current_position, relative_goal, nearest]))
        return np.array(observations, dtype=np.float32)

    def step(self, actions):
        # actions: [n_agents] discrete 0-6
        deltas = self.action_set

        for agent_id in range(self.n_agents):
            if self.done[agent_id] == 1:
                actions[agent_id] = 0


        next_positions = self.positions + deltas[actions]
        next_positions = np.clip(next_positions, 0, self.space_size - 1)

        # measure before move
        observations_before_move = self._observing_environment()

        # do move
        self.positions = next_positions

        # measure after move
        observations_after_move = self._observing_environment()

        rewards = np.zeros(self.n_agents)
        done = False

        # Reward for reaching goal
        for agent_id in range(self.n_agents):

            if self.done[agent_id] == 0 and np.allclose(self.positions[agent_id], self.goal[agent_id]):
                rewards[agent_id] += self.goal_reward
                self.done[agent_id] = 1
                if np.sum(self.done) == self.n_agents:
                    done = True
            else: # reward for approaching goal
                rewards[agent_id] += (np.linalg.norm(observations_before_move[agent_id][3:6]) - np.linalg.norm(observations_after_move[agent_id][3:6])) * self.distance_reward_factor
                # print(rewards[agent_id])

        # Penalty for collisions
        for agent_id in range(self.n_agents):
            for another_agent_id in range(agent_id + 1, self.n_agents):
                if np.array_equal(next_positions[agent_id], next_positions[another_agent_id]):
                    rewards[agent_id] += self.collision_reward
                    rewards[another_agent_id] += self.collision_reward


        return self._observing_environment(), rewards, done, {}