import time
# import os
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from SwarmEnvironment import *
from SwarmAgent import *
import numpy


# ------------------------------
# 4. Training Loop
# ------------------------------

numpy.random.seed(20)   # we need the random seed to be fixed so that other people can reproduce your exact training result
                        # so that they know you are not lying
manual_selected_device = "cuda"

n_agents = 10
visible_neighbor_amount = 1
space_size = 20
angular_displacement = 15
linear_displacement = 0.25
goal_error_tolerance = 1
collision_error_tolerance = 0.7
env = SwarmEnv(n_agents=n_agents, space_size=space_size, angular_displacement=angular_displacement,
               linear_displacement=linear_displacement, visible_neighbor_amount=visible_neighbor_amount)
env.set_random_goals()
agent = Agent(state_dim=env.observation_dimension - 3, action_dim=env.action_amount, max_coord=space_size,
              device=manual_selected_device)
print(env.action_amount)
agent.gamma = 0.95  # q learning gamma, learning rate
agent.epsilon = 1.0  # action randomness 1 for fully random
agent.batch_size = 128
agent.replay_buffer_size = 3000000

epsilon_decay = 0.995  # action randomness decay rate
epsilon_min = 0.05  # minimum epsilon

env.goal_reward = 200.0
env.collision_reward = -50.0
env.distance_reward_factor = 25.0
env.neighbor_approach_reward_factor = 10.0 # experimenting
env.step_reward = -0.2

training_steps = 2000
episodes_length = 256

total_rewards = np.zeros(training_steps)
epsilons = np.zeros(training_steps)
time_spend = np.zeros(training_steps)

total_time_start = time.time()
for episode in range(training_steps):
    observations = env.reset()
    total_reward = 0
    start_time = time.time()
    step = 0
    env.collision_count = 0
    for step in range(episodes_length):
        # Batched GPU/MPS inference for all agents
        actions = agent.select_multiple_actions(observations)  # replaces the for-loop

        # Environment step (expects actions as a list or array)
        next_observations, rewards, done, _ = env.step(actions, goal_error_tolerance=goal_error_tolerance,
                                                       collision_error_tolerance=collision_error_tolerance)

        # Store transitions for all agents
        for i in range(env.n_agents):
            agent.store(observations[i], actions[i], rewards[i], next_observations[i])

        # Train DQN
        agent.train_step()
        agent.update_target()  # gradual update target to prevent catastrophic forgetting

        # Move to next step
        observations = next_observations
        total_reward += np.mean(rewards)

        # End early if environment finishes
        if done:
            break

    delta_time = time.time() - start_time
    total_rewards[episode] = total_reward
    epsilons[episode] = delta_time
    time_spend[episode] = delta_time
    agent.epsilon = max(epsilon_min, agent.epsilon * epsilon_decay)
    # print(f"Episode {episode}, average total reward {total_reward:.2f}, eps {agent.epsilon:.2f}")

    print(
        f"Episode {episode + 1}, steps {step + 1:.0f} (done: {env.done_count / n_agents * 100.0:.3f}% "
        f"collision: {env.collision_count:.0f}), Average total reward {total_reward:.5f}, epsilon {agent.epsilon:.5f}, "
        f"replay buffer usage {float(len(agent.replay)) / float(agent.replay_buffer_size) * 100.0: .5f}%,"
        f"  time {delta_time:.5f}s")

total_time = time.time() - total_time_start
print(f"Total training time {total_time:.5f}s")
torch.save(agent.model.state_dict(), "swarm_agent_model.pth")
torch.save(agent.target.state_dict(), "swarm_target_model.pth")

