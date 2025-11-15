
import time

import torch
from SwarmEnvironment import *
from SwarmAgent import *
import numpy
from swarm_agent_visualization import *

load_model = True
numpy.random.seed(20)
manual_selected_device = torch.device("cuda")

n_agents = 20
space_size = 10
visible_neighbor_amount = 3
error_tolerance = 0.5 # to goal
collision_tolerance = 0.25
linear_displacement = 0.125


env = SwarmEnv(n_agents=n_agents, space_size=space_size, linear_displacement=linear_displacement, visible_neighbor_amount=visible_neighbor_amount)
env.set_random_goals()
agent = Agent(state_dim=env.observation_dimension, action_dim=env.action_amount, device=manual_selected_device)
if load_model:
    agent.model.load_state_dict(torch.load("swarm_agent_model.pth"))
    agent.target.load_state_dict(torch.load("swarm_target_model.pth"))
print(env.action_amount)

agent.gamma = 0.999 # q learning gamma
agent.epsilon = 0.0 # action randomness 1 for fully random
agent.batch_size = 128
agent.replay_buffer_size = 1000000

training_steps = 1024
episodes_length = 1024
save_flag = 128

epsilon_decay = 0.99 # action randomness decay rate

epsilon_min = 0.1 # minimum epsilon

env.non_goal_reward = -0.0 # not goood
env.stop_reward = -1.0
env.goal_reward = 20.0
env.collision_reward = -100.0
env.episode_reward = -0.0

env.distance_reward_factor = 2.0 / linear_displacement # how much nearest neighbor evey agent can visit

agent.model.eval()
agent.target.eval()

# env = SwarmEnv(n_agents=n_agents, space_size=space_size, linear_displacement=linear_displacement, visible_neighbor_amount=visible_neighbor_amount)
# env.set_random_goals()
# agent = Agent(state_dim=env.observation_dimension, action_dim=env.action_amount, device=manual_selected_device)



# agent.model.load_state_dict(torch.load("swarm_agent_model.pth"))
# agent.target.load_state_dict(torch.load("swarm_target_model.pth"))

visualize_swarm(agent, env, steps=500, save=True, error_tolerance=error_tolerance)