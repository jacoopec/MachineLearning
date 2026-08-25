import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Number of balls
N = 20

# Ball types: 0 or 1
types = np.random.randint(0, 2, N)

# Initial positions
positions = np.random.uniform(0, 10, (N, 2))

# Properties of the two types
colors = ["red", "blue"]
radii = [0.2, 0.4]

fig, ax = plt.subplots()


def update(frame):
    global positions

    # Move balls randomly
    positions += np.random.uniform(-0.2, 0.2, positions.shape)

    # Keep them inside the plot
    positions = np.clip(positions, 0, 10)

    # Clear the previous frame
    ax.clear()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    ax.set_title(f"Iteration {frame}")

    # Redraw every ball
    for i in range(N):
        ball_type = types[i]

        circle = Circle(
            positions[i],
            radius=radii[ball_type],
            color=colors[ball_type]
        )

        ax.add_patch(circle)


animation = FuncAnimation(
    fig,
    update,
    frames=200,
    interval=50
)

plt.show()