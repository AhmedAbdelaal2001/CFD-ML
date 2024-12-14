import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def initialize_domain(L, N, initial_condition_func):
    """
    Initializes the spatial domain and initial condition.

    Parameters:
    - L: Length of the domain.
    - N: Number of spatial grid points.
    - initial_condition_func: Function to set initial condition.

    Returns:
    - x: Spatial grid points.
    - u: Initial scalar quantity distribution.
    """
    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    u = initial_condition_func(x)
    return x, u

def apply_boundary_conditions(u, boundary_conditions):
    """
    Applies boundary conditions to the scalar field.

    Parameters:
    - u: Current scalar quantity distribution.
    - boundary_conditions: Tuple containing boundary values (u_left, u_right).

    Returns:
    - u: Updated scalar quantity distribution with boundary conditions applied.
    """
    u[0] = boundary_conditions[0]   # Left boundary
    u[-1] = boundary_conditions[1]  # Right boundary
    return u

def diffusion_step(u, alpha):
    """
    Performs a single time step update using the explicit FDM scheme.

    Parameters:
    - u: Current scalar quantity distribution.
    - alpha: Diffusion coefficient term (D*dt/dx^2).

    Returns:
    - u_new: Updated scalar quantity distribution after one time step.
    """
    u_new = u.copy()
    # Update internal points
    u_new[1:-1] = u[1:-1] + alpha * (u[2:] - 2*u[1:-1] + u[:-2])
    return u_new

def simulate_diffusion(L, N, T, D, initial_condition_func, boundary_conditions, 
                      save_animation=False, animation_filename='diffusion.gif'):
    """
    Simulates the diffusion process over time.

    Parameters:
    - L: Length of the domain.
    - N: Number of spatial grid points.
    - T: Total simulation time.
    - D: Diffusion coefficient.
    - initial_condition_func: Function to set initial condition.
    - boundary_conditions: Tuple containing boundary values (u_left, u_right).
    - save_animation: Boolean to save animation as GIF.
    - animation_filename: Filename for the saved animation.

    Returns:
    - x: Spatial grid points.
    - u_history: History of scalar quantity distributions over time.
    """
    x, u = initialize_domain(L, N, initial_condition_func)
    dx = L / (N - 1)
    
    # Time step based on stability criterion
    dt = 0.4 * dx**2 / D  # Safety factor < 0.5 for stability
    Nt = int(T / dt)
    alpha = D * dt / dx**2

    print(f"Simulation Parameters:")
    print(f"Domain Length (L): {L}")
    print(f"Number of Grid Points (N): {N}")
    print(f"Spatial Step (dx): {dx}")
    print(f"Time Step (dt): {dt}")
    print(f"Total Time Steps (Nt): {Nt}")
    print(f"Alpha (D*dt/dx^2): {alpha}")

    u_history = [u.copy()]

    for n in range(Nt):
        u = diffusion_step(u, alpha)
        u = apply_boundary_conditions(u, boundary_conditions)
        u_history.append(u.copy())
    
    u_history = np.array(u_history)
    return x, u_history, dt

def plot_results(x, u_history, dt, save_animation=False, animation_filename='diffusion.gif'):
    """
    Plots the diffusion process over time.

    Parameters:
    - x: Spatial grid points.
    - u_history: History of scalar quantity distributions over time.
    - dt: Time step size.
    - save_animation: Boolean to save animation as GIF.
    - animation_filename: Filename for the saved animation.
    """
    fig, ax = plt.subplots()
    line, = ax.plot(x, u_history[0], color='blue')
    ax.set_xlim(x.min() - x.min()/2, x.max() + x.max()/2)
    ax.set_ylim(np.min(u_history) - np.min(u_history)/2, np.max(u_history) + np.max(u_history)/2)
    ax.set_xlabel('Position')
    ax.set_ylabel('Concentration')
    ax.set_title('1D Diffusion Simulation')

    def update(frame):
        line.set_ydata(u_history[frame])
        ax.set_title(f'1D Diffusion at t = {frame*dt:.2f} s')
        return line,

    ani = FuncAnimation(fig, update, frames=range(0, len(u_history), max(1, len(u_history)//200)),
                        blit=True, interval=30)

    if save_animation:
        ani.save(animation_filename, writer='imagemagick', fps=30)
        print(f"Animation saved as {animation_filename}")
    else:
        plt.show()

def initial_condition_gaussian(x, x0=0.5, sigma=0.1, amplitude=1.0):
    """
    Sets a Gaussian initial condition.

    Parameters:
    - x: Spatial grid points.
    - x0: Mean of the Gaussian.
    - sigma: Standard deviation of the Gaussian.
    - amplitude: Peak amplitude of the Gaussian.

    Returns:
    - u: Initial scalar quantity distribution.
    """
    return amplitude * np.exp(-((x - x0)**2) / (2 * sigma**2))

def initial_condition_step(x, start, end):
    return np.where((x > start) & (x < end), 1.0, 0.0)

def main():
    # Simulation parameters
    L = 10            # Length of the domain
    N = 1000            # Number of spatial grid points
    T = 1            # Total simulation time
    D = 3            # Diffusion coefficient

    # Initial condition: Gaussian distribution
    initial_condition = lambda x: initial_condition_step(x, 1, 3)

    # Boundary conditions: u=0 at both ends (Dirichlet)
    boundary_conditions = (0, 0.5)

    # Run simulation
    x, u_history, dt = simulate_diffusion(L, N, T, D, initial_condition, boundary_conditions)

    # Plot results
    plot_results(x, u_history, dt, save_animation=False)

if __name__ == "__main__":
    main()
