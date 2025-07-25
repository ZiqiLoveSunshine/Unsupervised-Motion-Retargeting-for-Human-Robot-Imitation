import numpy as np
from matplotlib import pyplot as plt

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def pyplot_skeleton(topology, rel_positions, ax=None, show=True, color='black', relative = True):
    # Plot a skeleton in 3d
    if ax is None:
        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')

    positions = np.zeros_like(rel_positions)
    for j, i in enumerate(topology):
        if relative:
            positions[j] = rel_positions[j] + positions[i]
        else:
            positions[j] = rel_positions[j]
        ax.plot(
            np.array(positions[[i,j], 0]), 
            np.array(positions[[i,j], 1]), 
            np.array(positions[[i,j], 2]), 
            color=color
        )
    
    ax.scatter(
        np.array(positions[:, 0]), 
        np.array(positions[:, 1]), 
        np.array(positions[:, 2]), 
        color=color,
        marker='o'
    )
    if show:
        set_axes_equal(ax)
        plt.show()
    else:
        return ax