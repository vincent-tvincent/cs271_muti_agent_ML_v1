import time

import torch
from SwarmEnvironment import *
from SwarmAgent import *
import numpy
from swarm_agent_visualization import *

load_model = False
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
agent.epsilon = 1.0 # action randomness 1 for fully random
agent.batch_size = 128
agent.replay_buffer_size = 1000000

training_steps = 4096
episodes_length = 2048

epsilon_decay = 0.99 # action randomness decay rate

epsilon_min = 0.1 # minimum epsilon

env.non_goal_reward = -10.0
env.stop_reward = -1.0
env.goal_reward = 20.0
env.collision_reward = -100.0
env.episode_reward = -5.0

env.distance_reward_factor = 10.0 / linear_displacement # how much nearest neighbor evey agent can visit

total_rewards = np.zeros(training_steps)
epsilons = np.zeros(training_steps)
time_spend = np.zeros(training_steps)

# ------------------------------
# 4. Training Loop
# ------------------------------

for episode in range(training_steps):
    observations = env.reset()
    env.set_random_goals()
    total_reward = 0
    start_time = time.time()
    step = 0
    for step in range(episodes_length):
       # Batched GPU/MPS inference for all agents
       actions = agent.select_multiple_actions(observations)  # replaces the for-loop

        # Environment step (expects actions as a list or array)
       next_observations, rewards, done, _ = env.step(actions, error_tolerance=error_tolerance, collision_tolerance=collision_tolerance)

       # Store transitions for all agents
       for i in range(env.n_agents):
           agent.store(observations[i], actions[i], rewards[i], next_observations[i])

       # Train DQN
       agent.train_step()

       # Move to next step
       observations = next_observations
       total_reward += np.mean(rewards)

       # End early if environment finishes
       if done:
           # episodes_length = step + 1
           agent.train_step(done)
           break

    agent.update_target()
    # agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)

    delta_time = time.time() - start_time

    total_rewards[episode] = total_reward
    epsilons[episode] = agent.epsilon
    time_spend[episode] = delta_time
    print(f"Episode {episode + 1}, steps {step + 1:.0f} (done: {env.done_count/n_agents * 100.0:.3f}% collision: {env.collision_count:.0f}), Average total reward {total_reward:.5f}, epsilon {agent.epsilon:.5f} time {delta_time:.5f}s")

    agent.epsilon_decay(epsilon_min, epsilon_decay)

    if float(episode + 1) % (512.0) == 0 and step != 0:
        print("save checkpoint")
        torch.save(agent.model.state_dict(), "swarm_agent_model.pth")
        torch.save(agent.target.state_dict(), "swarm_target_model.pth")


# Save model weights
torch.save(agent.model.state_dict(), "swarm_agent_model.pth")
torch.save(agent.target.state_dict(), "swarm_target_model.pth")


# visualization
visualize_swarm(agent, env, steps=500, save=True, error_tolerance=error_tolerance)
