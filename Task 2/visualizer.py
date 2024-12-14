import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

class Visualizer:
    """
    Handles visualization of the 2D diffusion simulation results.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, U_history: np.ndarray, dt: float):
        """
        Initialize the visualizer.

        Parameters:
        - X, Y: 2D spatial grid points.
        - U_history: History of scalar quantity distributions over time as a 3D array.
        - dt: Time step size.
        """
        self.X = X
        self.Y = Y
        self.U_history = U_history
        self.dt = dt

    def plot(self, save_animation: bool=False, animation_filename: str='diffusion_2D.gif'):
        """
        Plot the 2D diffusion process over time.

        Parameters:
        - save_animation: Boolean to save animation as GIF.
        - animation_filename: Filename for the saved animation.
        """
        fig, ax = plt.subplots(figsize=(6,5))
        
        # Initial frame
        cax = ax.imshow(self.U_history[0], extent=(self.X.min(), self.X.max(), self.Y.min(), self.Y.max()),
                        origin='lower', cmap='hot', interpolation='nearest',
                        norm=colors.Normalize(vmin=np.min(self.U_history), vmax=np.max(self.U_history)))
        fig.colorbar(cax, ax=ax)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        title = ax.set_title('2D Diffusion at t = 0.00 s')

        def update(frame):
            cax.set_data(self.U_history[frame])
            title.set_text(f'2D Diffusion at t = {frame * self.dt:.2f} s')
            return cax, title

        # Determine frame step for animation smoothness and length
        max_frames = 200  # Maximum number of frames in the animation
        step = max(1, len(self.U_history) // max_frames)
        frames = range(0, len(self.U_history), step)

        ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=50, repeat=False)

        if save_animation:
            ani.save(animation_filename, writer='imagemagick', fps=30)
            print(f"Animation saved as {animation_filename}")
        else:
            plt.show()
