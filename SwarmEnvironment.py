import numpy as np

# ------------------------------
# 1. Simple 3D Swarm Environment
# ------------------------------
class SwarmEnv:
    def __init__(self, n_agents=5, space_size=10, linear_displacement=5.0, visible_neighbor_amount=10):

        self.n_agents = n_agents
        self.space_size = space_size

        # defaults values, ignore these for most case
        self.goal_reward = 1.0
        self.non_goal_reward = 0.0
        self.stop_reward = 0.0
        self.collision_reward = -1.0
        self.episode_reward = 0.0

        self.distance_reward_factor = 2.0


        self.visible_neighbor_amount = visible_neighbor_amount


        self.observation_dimension = 3 + 3 + self.visible_neighbor_amount * 3

        self.linear_displacement = linear_displacement
        self.action_set = self._generate_action_set(n=self.linear_displacement)
        # print(self.action_set)

        self.action_amount = self.action_set.shape[0]
        # print(self.action_amount)


        self.goal = np.zeros([self.n_agents, 3])
        self.done = np.zeros(self.n_agents)
        self.positions = np.zeros([self.n_agents, 3])
        self._observing_environment()
        self.reset()

    def _generate_action_set(self, n=1.0):
        return np.array([
            [0,0,0],
            [n,0,0],
            [0,n,0],
            [0,0,n],
            [-n,0,0],
            [0,-n,0],
            [0,0,-n],
        ])



    def set_random_goals(self):
        self.goal = np.random.uniform(0, self.space_size, size=self.goal.shape)

    def reset(self):
        self.collision_count = 0
        self.done_count = 0
        self.done = np.zeros(self.n_agents)
        self.positions = np.random.uniform(0, self.space_size, (self.n_agents, 3))
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

    def step(self, actions, error_tolerance=-1, collision_tolerance=-1):
        # actions: [n_agents] discrete 0-6
        deltas = self.action_set


        actions[self.done == 1] = 0

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
            rewards[agent_id] += self.episode_reward

            if error_tolerance <= 0:
                reach_goal = np.allclose(self.positions[agent_id], self.goal[agent_id])
            else:
                # print(np.linalg.norm(self.goal[agent_id] - self.positions[agent_id]))
                # reach_goal = np.linalg.norm(self.goal[agent_id] - self.positions[agent_id]) <= error_tolerance
                reach_goal = np.allclose(self.positions[agent_id], self.goal[agent_id], atol=error_tolerance)

            if reach_goal:

                if (self.done[agent_id] == 0):
                    rewards[agent_id] += self.goal_reward
                    # print(self.goal_reward)
                    self.done[agent_id] = 1
                self.done_count = np.sum(self.done)
                if self.done_count == self.n_agents:
                    done = True
            else: # reward for approaching goal and penalty for not at goal
                goal_vector_displacement = observations_before_move[agent_id][3:6] - observations_after_move[agent_id][
                    3:6]
                reference_unit_vector = observations_before_move[agent_id][3:6] / (
                        np.linalg.norm(observations_before_move[agent_id][3:6]) + 1e-8)
                progress_value = np.dot(reference_unit_vector, goal_vector_displacement)

                progress_value = progress_value if progress_value < 0.0 else 0.0
                rewards[agent_id] += progress_value * self.distance_reward_factor
                # rewards[agent_id] += np.mean(displacement_vector) * self.distance_reward_factor

                rewards[agent_id] += self.non_goal_reward
                if (actions[agent_id] == 0): rewards[agent_id] += self.stop_reward
                # print(rewards[agent_id])




        # Penalty for collisions
        for agent_id in range(self.n_agents):
            for another_agent_id in range(agent_id + 1, self.n_agents):
                if collision_tolerance <= 0:
                    collision_occured = np.allclose(self.positions[agent_id], self.positions[another_agent_id])
                else:
                    # collision_occured = np.abs(np.linalg.norm(self.positions[agent_id] - self.positions[another_agent_id])) <= collision_tolerance
                    collision_occured = np.allclose(self.positions[agent_id], self.goal[agent_id], atol=collision_tolerance)
                    # print(np.linalg.norm(self.positions[agent_id]) - np.linalg.norm(self.positions[another_agent_id]))
                if collision_occured:
                    self.collision_count += 1
                    rewards[agent_id] += self.collision_reward
                    rewards[another_agent_id] += self.collision_reward

        return self._observing_environment(), rewards, done, {}