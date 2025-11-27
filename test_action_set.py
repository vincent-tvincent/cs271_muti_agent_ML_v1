from SwarmEnvironment import SwarmEnv
import numpy as np

# Create a small environment just to generate the actions
env = SwarmEnv(
    n_agents=2,        # <= FIX
    space_size=10,
    angular_displacement=15,
    linear_displacement=1.0,
    visible_neighbor_amount=1
)

actions = env.action_set

print("Total actions:", len(actions))

# Check for uniqueness after rounding
rounded_actions = np.round(actions, 3)
unique_actions = np.unique(rounded_actions, axis=0)
print("Unique actions:", len(unique_actions))

# Print first few actions
print("First 10 actions:")
print(unique_actions[:10])

# Check for zero-length vectors (bad)
norms = np.linalg.norm(unique_actions, axis=1)
print("Zero-length vectors:", np.sum(norms == 0))

# Check if any action has large duplicate counts
_, idx, counts = np.unique(rounded_actions, axis=0, return_index=True, return_counts=True)
print("Max duplicates found (should be 1):", counts.max())
