import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

def initialize_domain_2D(Lx, Ly, Nx, Ny, initial_condition_func):
    """
    Initializes the 2D spatial domain and initial condition.

    Parameters:
    - Lx: Length of the domain in x-direction.
    - Ly: Length of the domain in y-direction.
    - Nx: Number of spatial grid points in x-direction.
    - Ny: Number of spatial grid points in y-direction.
    - initial_condition_func: Function to set initial condition, takes (X, Y) as input.

    Returns:
    - X, Y: 2D spatial grid points.
    - U: Initial scalar quantity distribution as a 2D array.
    - dx, dy: Spatial step sizes in x and y directions.
    """
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    U = initial_condition_func(X, Y)
    return X, Y, U, dx, dy

def apply_boundary_conditions_2D(U, boundary_conditions):
    """
    Applies Dirichlet boundary conditions to the 2D scalar field.

    Parameters:
    - U: Current scalar quantity distribution as a 2D array.
    - boundary_conditions: Dictionary with keys 'left', 'right', 'top', 'bottom' specifying boundary values.

    Returns:
    - U: Updated scalar quantity distribution with boundary conditions applied.
    """
    U[:, 0] = boundary_conditions.get('left', 0)     # Left boundary
    U[:, -1] = boundary_conditions.get('right', 0)   # Right boundary
    U[0, :] = boundary_conditions.get('top', 0)      # Top boundary
    U[-1, :] = boundary_conditions.get('bottom', 0)  # Bottom boundary
    return U

def diffusion_step_2D(U, alpha, beta):
    """
    Performs a single time step update using the explicit FDM scheme for 2D.

    Parameters:
    - U: Current scalar quantity distribution as a 2D array.
    - alpha: Diffusion coefficient term in x-direction (D*dt/dx^2).
    - beta: Diffusion coefficient term in y-direction (D*dt/dy^2).

    Returns:
    - U_new: Updated scalar quantity distribution after one time step.
    """
    U_new = U.copy()
    # Update internal points using vectorized operations
    U_new[1:-1,1:-1] = U[1:-1,1:-1] + \
                        alpha * (U[1:-1, 2:] - 2 * U[1:-1,1:-1] + U[1:-1, :-2]) + \
                        beta * (U[2:,1:-1] - 2 * U[1:-1,1:-1] + U[:-2,1:-1])
    return U_new

def simulate_diffusion_2D(Lx, Ly, Nx, Ny, T, D, initial_condition_func, boundary_conditions, 
                          save_animation=False, animation_filename='diffusion_2D.gif'):
    """
    Simulates the 2D diffusion process over time.

    Parameters:
    - Lx: Length of the domain in x-direction.
    - Ly: Length of the domain in y-direction.
    - Nx: Number of spatial grid points in x-direction.
    - Ny: Number of spatial grid points in y-direction.
    - T: Total simulation time.
    - D: Diffusion coefficient.
    - initial_condition_func: Function to set initial condition, takes (X, Y) as input.
    - boundary_conditions: Dictionary with keys 'left', 'right', 'top', 'bottom' specifying boundary values.
    - save_animation: Boolean to save animation as GIF.
    - animation_filename: Filename for the saved animation.

    Returns:
    - X, Y: 2D spatial grid points.
    - U_history: History of scalar quantity distributions over time as a 3D array.
    - dt: Time step size.
    """
    X, Y, U, dx, dy = initialize_domain_2D(Lx, Ly, Nx, Ny, initial_condition_func)
    
    # Time step based on stability criterion for 2D: alpha + beta <= 0.5
    # For simplicity, assume dx = dy, hence alpha = beta
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    dt = 0.25 * min(dx, dy)**2 / D  # Safety factor < 0.25 for 2D stability
    Nt = int(T / dt)
    alpha = D * dt / dx**2
    beta = D * dt / dy**2

    print(f"Simulation Parameters:")
    print(f"Domain Lengths (Lx, Ly): ({Lx}, {Ly})")
    print(f"Number of Grid Points (Nx, Ny): ({Nx}, {Ny})")
    print(f"Spatial Steps (dx, dy): ({dx}, {dy})")
    print(f"Time Step (dt): {dt}")
    print(f"Total Time Steps (Nt): {Nt}")
    print(f"Alpha (D*dt/dx^2): {alpha}")
    print(f"Beta (D*dt/dy^2): {beta}")

    # Initialize history list
    U_history = [U.copy()]

    for n in range(Nt):
        U = diffusion_step_2D(U, alpha, beta)
        U = apply_boundary_conditions_2D(U, boundary_conditions)
        U_history.append(U.copy())
        if n % (Nt // 10) == 0:
            print(f"Progress: {n}/{Nt} steps completed.")

    U_history = np.array(U_history)
    return X, Y, U_history, dt

def plot_results_2D(X, Y, U_history, dt, save_animation=False, animation_filename='diffusion_2D.gif'):
    """
    Plots the 2D diffusion process over time.

    Parameters:
    - X, Y: 2D spatial grid points.
    - U_history: History of scalar quantity distributions over time as a 3D array.
    - dt: Time step size.
    - save_animation: Boolean to save animation as GIF.
    - animation_filename: Filename for the saved animation.
    """
    fig, ax = plt.subplots(figsize=(6,5))
    
    # Initial frame
    cax = ax.imshow(U_history[0], extent=(X.min(), X.max(), Y.min(), Y.max()),
                    origin='lower', cmap='hot', interpolation='nearest',
                    norm=colors.Normalize(vmin=np.min(U_history), vmax=np.max(U_history)))
    fig.colorbar(cax, ax=ax)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    title = ax.set_title('2D Diffusion at t = 0.00 s')

    def update(frame):
        cax.set_data(U_history[frame])
        title.set_text(f'2D Diffusion at t = {frame*dt:.2f} s')
        return cax, title

    # Determine frame step for animation smoothness and length
    max_frames = 200  # Maximum number of frames in the animation
    step = max(1, len(U_history) // max_frames)
    frames = range(0, len(U_history), step)

    ani = FuncAnimation(fig, update, frames=frames, blit=False, interval=50, repeat=False)

    if save_animation:
        ani.save(animation_filename, writer='imagemagick', fps=30)
        print(f"Animation saved as {animation_filename}")
    else:
        plt.show()

def initial_condition_gaussian_2D(X, Y, x0=5, y0=5, sigma=1.0, amplitude=1.0):
    """
    Sets a Gaussian initial condition in 2D.

    Parameters:
    - X, Y: 2D spatial grid points.
    - x0, y0: Center of the Gaussian.
    - sigma: Standard deviation of the Gaussian.
    - amplitude: Peak amplitude of the Gaussian.

    Returns:
    - U: Initial scalar quantity distribution as a 2D array.
    """
    return amplitude * np.exp(-(((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)))

def initial_condition_step_2D(X, Y, start_x, end_x, start_y, end_y, value=1.0):
    """
    Sets a square step function initial condition in 2D.

    Parameters:
    - X, Y: 2D spatial grid points.
    - start_x, end_x: Start and end positions in x-direction for the step.
    - start_y, end_y: Start and end positions in y-direction for the step.
    - value: Value inside the step.

    Returns:
    - U: Initial scalar quantity distribution as a 2D array.
    """
    return np.where((X > start_x) & (X < end_x) & (Y > start_y) & (Y < end_y), value, 0.0)

def initial_condition_circular_hotspot_2D(X, Y, x_center, y_center, radius, hot_value=5.0, base_value=0.0):
    """
    Sets a circular hotspot initial condition in 2D.

    Parameters:
    - X, Y: 2D spatial grid points.
    - x_center, y_center: Coordinates of the hotspot center.
    - radius: Radius of the hotspot.
    - hot_value: Scalar quantity value inside the hotspot.
    - base_value: Scalar quantity value outside the hotspot.

    Returns:
    - U: Initial scalar quantity distribution as a 2D array.
    """
    distance_squared = (X - x_center)**2 + (Y - y_center)**2
    U = np.where(distance_squared <= radius**2, hot_value, base_value)
    return U


def main():
    # Simulation parameters
    Lx = 10.0            # Length of the domain in x-direction
    Ly = 10.0            # Length of the domain in y-direction
    Nx = 100             # Number of spatial grid points in x-direction
    Ny = 100             # Number of spatial grid points in y-direction
    T = 5.0              # Total simulation time
    D = 1.0              # Diffusion coefficient

    # Initial condition: Gaussian distribution
    initial_condition = lambda X, Y: initial_condition_circular_hotspot_2D(X, Y, 5, 5, 1)
    
    # Alternatively, use a square step initial condition
    # initial_condition = lambda X, Y: initial_condition_step_2D(X, Y, start_x=4, end_x=6, start_y=4, end_y=6, value=5.0)

    # Boundary conditions: u=0 on all boundaries (Dirichlet)
    boundary_conditions = {
        'left': 1.0,
        'right': 0.0,
        'top': 1.0,
        'bottom': 0.0
    }

    # Run simulation
    X, Y, U_history, dt = simulate_diffusion_2D(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        T=T,
        D=D,
        initial_condition_func=initial_condition,
        boundary_conditions=boundary_conditions,
        save_animation=False  # Set to True to save the animation
    )

    # Plot results
    plot_results_2D(
        X=X,
        Y=Y,
        U_history=U_history,
        dt=dt,
        save_animation=False,  # Set to True to save the animation
        animation_filename='diffusion_2D.gif'
    )

if __name__ == "__main__":
    main()
