
import time

import torch
from SwarmEnvironment import *
from SwarmAgent import *
import numpy
from swarm_agent_visualization import *

load_model = False
numpy.random.seed(20)
manual_selected_device = torch.device("cuda")

n_agents = 4
space_size = 10
visible_neighbor_amount = 3
error_tolerance = 0.5 # to goal
collision_tolerance = 0.25
linear_displacement = 0.125


env = SwarmEnv(n_agents=n_agents, space_size=space_size, linear_displacement=linear_displacement, visible_neighbor_amount=visible_neighbor_amount)
env.set_random_goals()
agent = Agent(state_dim=env.observation_dimension, action_dim=env.action_amount, device=manual_selected_device)

agent.model.load_state_dict(torch.load("swarm_agent_model.pth"))
agent.target.load_state_dict(torch.load("swarm_target_model.pth"))

visualize_swarm(agent, env, steps=2000, save=True, error_tolerance=error_tolerance)