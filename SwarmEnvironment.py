import numpy as np

# ------------------------------
# 1. Simple 3D Swarm Environment
# ------------------------------
class SwarmEnv:
    def __init__(self, n_agents=5, space_size=10, angular_displacement=22.5, linear_displacement=5.0, visible_neighbor_amount=10):


        self.n_agents = n_agents
        self.space_size = space_size

        # defaults values, ignore these for most case
        self.goal_reward = 1.0
        self.collision_reward = -1.0
        self.distance_reward_factor = 2.0
        self.visible_neighbor_amount = visible_neighbor_amount

        self.observation_dimension = 3 + 3 + self.visible_neighbor_amount * 3


        self.angular_displacement = angular_displacement
        self.linear_displacement = linear_displacement
        self.action_set = self._generate_action_set(n=self.linear_displacement, d=self.angular_displacement, degrees=True)
        # print(self.action_set)

        self.action_amount = self.action_set.shape[0]
        # print(self.action_amount)


        self.goal = np.zeros([self.n_agents, 3])
        self.done = np.zeros(self.n_agents)
        self.positions = np.zeros([self.n_agents, 3])
        self._observing_environment()
        self.reset()

    def _generate_action_set(self, n=1.0, d=5.0, degrees=True):

        if degrees:
            d = np.deg2rad(d)

        theta_vals = np.arange(0, 2 * np.pi, d)
        phi_vals = np.arange(0, np.pi + d, d)

        actions = []

        v = 0.0
        for i in range(10):
            v += n / 10.0
            for phi in phi_vals:
                for theta in theta_vals:
                    # Convert spherical to Cartesian
                    x = n * np.sin(phi) * np.cos(theta)
                    y = n * np.sin(phi) * np.sin(theta)
                    z = n * np.cos(phi)
                    actions.append([x, y, z])
        return np.array(actions)


    def set_random_goals(self):
        self.goal = np.random.randint(0, self.space_size, size=self.goal.shape)

    def reset(self):
        self.done = np.zeros(self.n_agents)
        self.positions = np.random.randint(0, self.space_size, (self.n_agents, 3))
        return self._observing_environment()

    def _observing_environment(self):
        # Each agent observes its position + relative goal + nearest neighbor
        observations = []
        for agent_id in range(self.n_agents):
            current_position = self.positions[agent_id]
            relative_goal = self.goal[agent_id] - current_position
            distances = np.linalg.norm(self.positions - current_position, axis=1)
            nearest = self.positions[np.argsort(distances)[1:self.visible_neighbor_amount + 1]] - current_position  # skip itself
            observations.append(np.concatenate([current_position, relative_goal, *nearest]))
        return np.array(observations, dtype=np.float32)

    def step(self, actions, error_tolerance=-1):
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

            if error_tolerance <= 0:
                reach_goal = np.allclose(self.positions[agent_id], self.goal[agent_id])
            else:
                reach_goal = np.allclose(self.positions[agent_id], self.goal[agent_id], atol=error_tolerance)

            if self.done[agent_id] == 0 and reach_goal:
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
                if error_tolerance <= 0:
                    collision_occured = np.allclose(self.positions[agent_id], self.positions[another_agent_id])
                else:
                    collision_occured = np.allclose(self.positions[agent_id], self.positions[another_agent_id], atol=error_tolerance)
                if collision_occured:
                    rewards[agent_id] += self.collision_reward
                    rewards[another_agent_id] += self.collision_reward

        return self._observing_environment(), rewards, done, {}