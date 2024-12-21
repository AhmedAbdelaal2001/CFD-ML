# visualizer.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors
import os  # Import os module for directory operations

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

    def plot(self):
        """
        Plot the 2D diffusion process over time and save snapshots.

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
        plt.show()

        # --- Added Functionality: Save Snapshots ---
        self.save_snapshots(results_dir='results')
        # --------------------------------------------

    def save_snapshots(self, results_dir: str='results', snapshot_step: int=20):
        """
        Save each timestep as an image in the specified directory at specified intervals.

        Parameters:
        - results_dir: Directory to save snapshot images.
        - snapshot_step: Interval for saving snapshots. Save every 'snapshot_step' frames.
        """
        # Create the results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        print(f"Saving snapshots to the '{results_dir}' directory...")

        # Pre-create figure and axis to reuse
        fig, ax = plt.subplots(figsize=(6,5))
        vmin = np.min(self.U_history)
        vmax = np.max(self.U_history)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cax = ax.imshow(self.U_history[0], extent=(self.X.min(), self.X.max(), self.Y.min(), self.Y.max()),
                        origin='lower', cmap='hot', interpolation='nearest', norm=norm)
        fig.colorbar(cax, ax=ax)
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        title = ax.set_title('2D Diffusion at t = 0.00 s')

        for i, U in enumerate(self.U_history):
            if i % snapshot_step != 0:
                continue  # Skip frames that are not at the snapshot step

            # Update data and title
            cax.set_data(U)
            title.set_text(f'2D Diffusion at t = {i * self.dt:.2f} s')

            # Define the filename with zero-padded indexing
            filename = os.path.join(results_dir, f'snapshot_{i:04d}.png')

            # Save the figure
            fig.savefig(filename)
            
            # Print progress every 10 snapshots
            if (i + 1) % (snapshot_step * 10) == 0 or (i + 1) == len(self.U_history):
                print(f"Saved snapshot {i + 1}/{len(self.U_history)} to '{filename}'")

        plt.close(fig)  # Close the figure after saving all snapshots
        print(f"All snapshots have been saved in the '{results_dir}' directory.")

