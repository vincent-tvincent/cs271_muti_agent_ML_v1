import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.display import display, clear_output

def visualize_swarm(agent, env, steps=50, save=False, goal_error_tolerance = 1, collision_error_tolerance = 1, interval=10):
    """
    Visualize swarm movement in 3D and optionally save 4 views as .gif.
    - save=False → real-time live animation (in Jupyter)
    - save=True  → export 4 gifs: normal, xy, xz, yz
    """
    #np.random.seed(20)
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0
    obs = env.reset()
    positions_history = [env.positions.copy()]
    print("stepping")
    for _ in range(steps):
        actions = [agent.select_action(o) for o in obs]
        obs, _, done, _ = env.step(actions, goal_error_tolerance=goal_error_tolerance, collision_error_tolerance=collision_error_tolerance)
        positions_history.append(env.positions.copy())
        if done:
            break
    print("stepping complete")
    print("plotting")
    positions_history = np.array(positions_history)  # shape: [T, n_agents, 3]
    n_steps, n_agents, _ = positions_history.shape

    # Normalize goal shape
    goals = np.atleast_2d(env.goal)
    if goals.shape[0] == 1:
        goals = np.repeat(goals, n_agents, axis=0)

    # ----------------------------
    # Helper to make and save GIFs
    # ----------------------------
    def make_animation(view_name, elev, azim):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, env.space_size)
        ax.set_ylim(0, env.space_size)
        ax.set_zlim(0, env.space_size)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"3D Swarm Movement ({view_name})")

        scat = ax.scatter([], [], [], c='blue', s=50, label='Agents')
        ax.scatter(goals[:,0], goals[:,1], goals[:,2],
                   c='red', s=100, marker='*', label='Goals')

        lines = [ax.plot([], [], [], 'gray', linestyle='--', linewidth=1)[0]
                 for _ in range(n_agents)]

        ax.view_init(elev=elev, azim=azim)
        ax.legend()

        def init():
            scat._offsets3d = ([], [], [])
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            return [scat, *lines]

        def update(frame):
            pos = positions_history[frame]
            scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
            for i, line in enumerate(lines):
                x = [pos[i, 0], goals[i, 0]]
                y = [pos[i, 1], goals[i, 1]]
                z = [pos[i, 2], goals[i, 2]]
                line.set_data(x, y)
                line.set_3d_properties(z)
            ax.set_title(f"3D Swarm Movement ({view_name}) - Step {frame}/{n_steps}")
            return [scat, *lines]

        ani = animation.FuncAnimation(
            fig, update, frames=n_steps, init_func=init,
            interval=interval, blit=False
        )

        if save:
            filename = f"swarm_simulation_{view_name.lower()}.gif"
            ani.save(filename, writer='pillow')
            print(f"✅ Saved {filename}")
        else:
            plt.show()

        plt.close(fig)


    # ------------------------------------------------
    # A) SAVE FOUR VIEWS AS GIFS
    # ------------------------------------------------
    if save:
        make_animation("normal", elev=30, azim=45)
        make_animation("xy", elev=90, azim=-90)
        make_animation("xz", elev=0, azim=-90)
        make_animation("yz", elev=0, azim=0)
        return

    # ------------------------------------------------
    # B) LIVE DISPLAY (REAL-TIME UPDATE in Jupyter)
    # ------------------------------------------------
    plt.ion()  # Turn on interactive mode
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, env.space_size)
    ax.set_ylim(0, env.space_size)
    ax.set_zlim(0, env.space_size)
    ax.set_title("3D Swarm Movement (Live)")
    scat = ax.scatter([], [], [], c='blue', s=50, label='Agents')
    ax.scatter(goals[:,0], goals[:,1], goals[:,2],
               c='red', s=100, marker='*', label='Goals')
    lines = [ax.plot([], [], [], 'gray', linestyle='--', linewidth=1)[0]
             for _ in range(n_agents)]
    ax.legend()

    for frame in range(n_steps):
        pos = positions_history[frame]
        scat._offsets3d = (pos[:,0], pos[:,1], pos[:,2])
        for i, line in enumerate(lines):
            x = [pos[i, 0], goals[i, 0]]
            y = [pos[i, 1], goals[i, 1]]
            z = [pos[i, 2], goals[i, 2]]
            line.set_data(x, y)
            line.set_3d_properties(z)
        ax.set_title(f"3D Swarm Movement (Live) - Step {frame}/{n_steps}")
        display(fig)
        clear_output(wait=True)
        plt.pause(interval / 1000.0)

    plt.ioff()
    plt.show()

    agent.epsilon = original_epsilon
