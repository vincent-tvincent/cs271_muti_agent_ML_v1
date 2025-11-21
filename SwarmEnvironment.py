import numpy as np
from scipy.spatial.distance import cdist
print("hello")

# ------------------------------
# 1. Simple 3D Swarm Environment
# ------------------------------
class SwarmEnv:
    def __init__(self, n_agents=5, space_size=10, angular_displacement=22.5, linear_displacement=1.0, visible_neighbor_amount=10):


        self.n_agents = n_agents
        self.space_size = space_size

        # defaults values, ignore these for most case
        self.goal_reward = 1.0
        self.collision_reward = -1.0
        self.distance_reward_factor = 2.0
        self.neighbor_approach_reward_factor = 0.5
        self.visible_neighbor_amount = visible_neighbor_amount
        self.step_reward = -1.0

        self.observation_dimension = 3 + 3 + self.visible_neighbor_amount * 3


        self.angular_displacement = angular_displacement
        self.linear_displacement = linear_displacement
        self.action_set = self._generate_action_set(n=self.linear_displacement, d=self.angular_displacement, degrees=True)
        # print(self.action_set)

        self.action_amount = self.action_set.shape[0]
        #print(self.action_amount)


        self.goal = np.zeros([self.n_agents, 3])
        self.done = np.zeros(self.n_agents)
        self.positions = np.zeros([self.n_agents, 3])
        self._observing_environment()
        self.reset()

    def _generate_action_set(self, n=1.0, d=22.5, degrees=True):

        if degrees:
            d = np.deg2rad(d)

        theta_vals = np.arange(0, 2 * np.pi, d)
        phi_vals = np.arange(0, np.pi + d, d)

        actions = []
        actions.append([0,0,0]) #action at 0 index is the "do nothing" action
        
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
        #np.random.seed(20)
        self.collision_count = 0
        self.done_count = 0
        self.done = np.zeros(self.n_agents)
        self.positions = np.random.randint(0, self.space_size, (self.n_agents, 3))
        self.set_random_goals()
        return self._observing_environment()

    def _observing_environment(self):
        # Each agent observes its position + relative goal + nearest neighbor
        observations = np.zeros([self.n_agents, self.observation_dimension])

        observations[:, 0:3] = self.positions
        observations[:, 3:6] = self.goal - self.positions

        for agent_id in range(self.n_agents):
            # current_position = self.positions[agent_id]
            # relative_goal = self.goal[agent_id] - current_position
            distances = np.linalg.norm(self.positions - self.positions[agent_id], axis=1)
            nearest = self.positions[np.argsort(distances)[1:self.visible_neighbor_amount + 1]] - self.positions[agent_id]  # skip itself
            # observations.append(np.concatenate([current_position, relative_goal, *nearest]))
            observations[agent_id, 6:self.observation_dimension] = nearest.flatten()
        return np.array(observations, dtype=np.float32)

    def step(self, actions, goal_error_tolerance=-1, collision_error_tolerance=-1):
        # actions: [n_agents] discrete
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
        

        # Vectorized rewards process, for more efficiency:
        #check which agents are at the goal currently
        if goal_error_tolerance <= 0:
            is_at_goal = np.all(self.positions == self.goal, axis=1)
        else:
            is_close_per_dim = np.abs(self.positions - self.goal) <= goal_error_tolerance
            is_at_goal = np.all(is_close_per_dim, axis=1)

        #create a mask for agents that were not done before this step and are now done
        newly_done_mask = (1-self.done.astype(int)) * is_at_goal

        #apply goal reward using the mask
        rewards += newly_done_mask * self.goal_reward

        #update the global self.done status
        self.done = np.logical_or(self.done, newly_done_mask)
        self.done_count = np.sum(self.done)

        #create a mask for agents that are still active (not done)
        active_agents_mask = (1 - self.done.astype(int))

        #calculate distance reward for all agents
        norm_before = np.linalg.norm(observations_before_move[:, 3:6], axis=1)
        norm_after = np.linalg.norm(observations_after_move[:, 3:6], axis=1)
        distance_rewards = (norm_before - norm_after) * self.distance_reward_factor

        #calculate distance reward for approaching or leave the nearest neighbor
        neighbors_norm_before = np.linalg.norm(observations_before_move[:, 6:9], axis=1)
        neighbors_norm_after = np.linalg.norm(observations_after_move[:, 6:9], axis=1)
        approach_neighbor_rewards = (neighbors_norm_after - neighbors_norm_before) * self.neighbor_approach_reward_factor
        approach_neighbor_rewards *= approach_neighbor_rewards <= 0

        #apply distance reward only to active agents
        rewards += (
                           distance_rewards +
                           approach_neighbor_rewards +
                           self.step_reward
                    ) * active_agents_mask

        # #apply step penalty only to active agents
        # rewards += self.step_reward * active_agents_mask

        #check global done status
        done = np.sum(self.done) == self.n_agents

        #collision penalty using distance matrix, more efficient method
        distance_matrix = cdist(next_positions, next_positions)

        #identify all agent pairs that are close enough to collide, ignoring diagonal (distance to self is 0)
        collision_matrix = (distance_matrix <= collision_error_tolerance) & (distance_matrix > 0)

        #sum the collisions for each agent
        n_collisions = np.sum(collision_matrix, axis=1)

        actual_n_collisions = n_collisions * active_agents_mask #agents that have finished receive no rewards/punishments
        self.collision_count += np.sum(actual_n_collisions)/2
        rewards += actual_n_collisions * self.collision_reward


        return self._observing_environment(), rewards, done, {}