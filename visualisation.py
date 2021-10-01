import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

def animate(history):
    frames = len(history)
    print("Rendering {} frames".format(frames))
    fig = plt.figure(figsize=(6, 2))
    fig_grid = fig.add_subplot(121)
    fig_energy = fig.add_subplot(243)
    fig_visible = fig.add_subplot(244)
    fig_energy.set_autoscale_on(False)
    energy_plt = np.zeros((frames, 1))

    def render_frame(i):
        grid, visible, energy = history[i]
        # Render grid
        fig_grid.matshow(visible, vmin=-1, vmax=1, cmap='jet')
        # Render energy chart
        energy_plt[i] = energy
        fig_energy.clear()
        fig_energy.axis([0, frames, 0, 2])
        fig_energy.plot(energy_plt[:i + 1])

    an = animation.FuncAnimation(
        fig, render_frame, frames=frames, interval=100
    )

    plt.show()