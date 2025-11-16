
import time

import torch
from SwarmEnvironment import *
from SwarmAgent import *
import numpy
from swarm_agent_visualization import *

load_model = True
numpy.random.seed(20)
manual_selected_device = torch.device("cuda")

n_agents = 10
space_size = 20
visible_neighbor_amount = 1
goal_error_tolerance = 1 # to goal
collision_error_tolerance = 0.7
linear_displacement = 0.5


env = SwarmEnv(n_agents=n_agents, space_size=space_size, linear_displacement=linear_displacement, visible_neighbor_amount=visible_neighbor_amount)
env.set_random_goals()
agent = Agent(state_dim=env.observation_dimension, action_dim=env.action_amount, device=manual_selected_device)
if load_model:
    agent.model.load_state_dict(torch.load("swarm_agent_model.pth"))
    agent.target.load_state_dict(torch.load("swarm_target_model.pth"))
print(env.action_amount)

agent.gamma = 0.95 # q learning gamma
agent.epsilon = 0.0 # action randomness 1 for fully random
agent.batch_size = 64
agent.replay_buffer_size = 10000

training_steps = 1000
episodes_length = 300
save_flag = 128

epsilon_decay = 0.995 # action randomness decay rate

epsilon_min = 0.05 # minimum epsilon

env.goal_reward = 100.0
env.collision_reward = -8.0
env.step_reward = -0.2

env.distance_reward_factor = 25 # how much nearest neighbor evey agent can visit

agent.model.eval()
agent.target.eval()

# env = SwarmEnv(n_agents=n_agents, space_size=space_size, linear_displacement=linear_displacement, visible_neighbor_amount=visible_neighbor_amount)
# env.set_random_goals()
# agent = Agent(state_dim=env.observation_dimension, action_dim=env.action_amount, device=manual_selected_device)



# agent.model.load_state_dict(torch.load("swarm_agent_model.pth"))
# agent.target.load_state_dict(torch.load("swarm_target_model.pth"))

visualize_swarm(agent, env, steps=500, save=True, goal_error_tolerance=goal_error_tolerance, collision_error_tolerance = collision_error_tolerance)