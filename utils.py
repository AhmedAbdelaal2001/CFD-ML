import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as colors

# Define abstract base class for regions
class Region:
    def apply(self, X, Y):
        """
        Applies the region's scalar function to the grid.

        Parameters:
        - X, Y: 2D spatial grid points.

        Returns:
        - U_region: 2D array with scalar values for this region.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

# Rectangular region class
class RectangularRegion(Region):
    def __init__(self, start_x, end_x, start_y, end_y, scalar_func):
        """
        Initializes a rectangular region.

        Parameters:
        - start_x, end_x: Boundaries in the x-direction.
        - start_y, end_y: Boundaries in the y-direction.
        - scalar_func: Function f(x, y) defining scalar values within the region.
        """
        self.start_x = start_x
        self.end_x = end_x
        self.start_y = start_y
        self.end_y = end_y
        self.scalar_func = scalar_func

    def apply(self, X, Y):
        mask = (X >= self.start_x) & (X <= self.end_x) & (Y >= self.start_y) & (Y <= self.end_y)
        U_region = np.where(mask, self.scalar_func(X, Y), 0.0)
        return U_region

# Elliptic region class
class EllipticRegion(Region):
    def __init__(self, x_center, y_center, a, b, scalar_func):
        """
        Initializes an elliptic region.

        Parameters:
        - x_center, y_center: Center coordinates of the ellipse.
        - a: Radius in the x-direction (semi-major axis).
        - b: Radius in the y-direction (semi-minor axis).
        - scalar_func: Function f(x, y) defining scalar values within the ellipse.
        """
        self.x_center = x_center
        self.y_center = y_center
        self.a = a
        self.b = b
        self.scalar_func = scalar_func

    def apply(self, X, Y):
        ellipse_eq = ((X - self.x_center) / self.a)**2 + ((Y - self.y_center) / self.b)**2
        mask = ellipse_eq <= 1
        U_region = np.where(mask, self.scalar_func(X, Y), 0.0)
        return U_region

def initialize_domain_2D(Lx, Ly, Nx, Ny):
    """
    Initializes the 2D spatial domain.

    Parameters:
    - Lx: Length of the domain in x-direction.
    - Ly: Length of the domain in y-direction.
    - Nx: Number of spatial grid points in x-direction.
    - Ny: Number of spatial grid points in y-direction.

    Returns:
    - X, Y: 2D spatial grid points.
    - dx, dy: Spatial step sizes in x and y directions.
    """
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    return X, Y, dx, dy

def initialize_multiple_regions(X, Y, regions):
    """
    Initializes the scalar field by applying multiple regions.

    Parameters:
    - X, Y: 2D spatial grid points.
    - regions: List of Region objects.

    Returns:
    - U: Initial scalar quantity distribution as a 2D array.
    """
    U = np.zeros_like(X)
    for region in regions:
        U += region.apply(X, Y)
    return U

def apply_boundary_conditions_2D(U, boundary_conditions, X, Y):
    """
    Applies boundary conditions to the 2D scalar field. Boundary conditions can be
    functions of a single spatial variable or constants.

    Parameters:
    - U: Current scalar quantity distribution as a 2D array.
    - boundary_conditions: Dictionary with keys 'left', 'right', 'top', 'bottom'.
                           Values can be functions or constants.
    - X, Y: 2D spatial grid points.

    Returns:
    - U: Updated scalar quantity distribution with boundary conditions applied.
    """
    # Left Boundary (x = 0), varies with y
    if 'left' in boundary_conditions:
        bc_left = boundary_conditions['left']
        if callable(bc_left):
            U[:, 0] = bc_left(Y[:, 0])
        else:
            U[:, 0] = bc_left
    else:
        U[:, 0] = 0.0  # Default value if not specified

    # Right Boundary (x = Lx), varies with y
    if 'right' in boundary_conditions:
        bc_right = boundary_conditions['right']
        if callable(bc_right):
            U[:, -1] = bc_right(Y[:, -1])
        else:
            U[:, -1] = bc_right
    else:
        U[:, -1] = 0.0  # Default value if not specified

    # Bottom Boundary (y = 0), varies with x
    if 'bottom' in boundary_conditions:
        bc_bottom = boundary_conditions['bottom']
        if callable(bc_bottom):
            U[0, :] = bc_bottom(X[0, :])
        else:
            U[0, :] = bc_bottom
    else:
        U[0, :] = 0.0  # Default value if not specified

    # Top Boundary (y = Ly), varies with x
    if 'top' in boundary_conditions:
        bc_top = boundary_conditions['top']
        if callable(bc_top):
            U[-1, :] = bc_top(X[-1, :])
        else:
            U[-1, :] = bc_top
    else:
        U[-1, :] = 0.0  # Default value if not specified

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

def simulate_diffusion_2D(Lx, Ly, Nx, Ny, T, D, regions, boundary_conditions, 
                          save_animation=False, animation_filename='diffusion_2D.gif'):
    """
    Simulates the 2D diffusion process over time with multiple initial regions.

    Parameters:
    - Lx: Length of the domain in x-direction.
    - Ly: Length of the domain in y-direction.
    - Nx: Number of spatial grid points in x-direction.
    - Ny: Number of spatial grid points in y-direction.
    - T: Total simulation time.
    - D: Diffusion coefficient.
    - regions: List of Region objects defining initial conditions.
    - boundary_conditions: Dictionary with keys 'left', 'right', 'top', 'bottom' specifying boundary functions or constants.
    - save_animation: Boolean to save animation as GIF.
    - animation_filename: Filename for the saved animation.

    Returns:
    - X, Y: 2D spatial grid points.
    - U_history: History of scalar quantity distributions over time as a 3D array.
    - dt: Time step size.
    """
    X, Y, dx, dy = initialize_domain_2D(Lx, Ly, Nx, Ny)
    U = initialize_multiple_regions(X, Y, regions)
    
    # Time step based on stability criterion for 2D: alpha + beta <= 0.5
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

    # Apply initial boundary conditions
    U = apply_boundary_conditions_2D(U, boundary_conditions, X, Y)

    # Initialize history list
    U_history = [U.copy()]

    for n in range(Nt):
        U = diffusion_step_2D(U, alpha, beta)
        U = apply_boundary_conditions_2D(U, boundary_conditions, X, Y)
        U_history.append(U.copy())
        if n % max(1, (Nt // 10)) == 0:
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

def main():
    # Simulation parameters
    Lx = 10.0            # Length of the domain in x-direction
    Ly = 10.0            # Length of the domain in y-direction
    Nx = 100             # Number of spatial grid points in x-direction
    Ny = 100             # Number of spatial grid points in y-direction
    T = 5.0              # Total simulation time
    D = 1.0              # Diffusion coefficient

    # Define regions
    regions = []

    # Example Region 1: Rectangular region with a constant value
    def rect_scalar_func(X, Y):
        return 5.0  # Constant value inside the rectangle

    rect_region = RectangularRegion(
        start_x=2.0, end_x=4.0,
        start_y=2.0, end_y=4.0,
        scalar_func=rect_scalar_func
    )
    regions.append(rect_region)

    # Example Region 2: Elliptic region with a Gaussian distribution
    def ellipse_scalar_func(X, Y):
        x0, y0 = 7.0, 7.0  # Center of the ellipse
        sigma = 0.5
        return 3.0 * np.exp(-(((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)))

    ellipse_region = EllipticRegion(
        x_center=7.0, y_center=7.0,
        a=1.0, b=1.5,
        scalar_func=ellipse_scalar_func
    )
    regions.append(ellipse_region)

    # Example Region 3: Overlapping Elliptic region with different scalar function
    def overlapping_ellipse_scalar_func(X, Y):
        return 2.0 * np.ones_like(X)  # Constant value inside

    overlapping_ellipse_region = EllipticRegion(
        x_center=3.0, y_center=7.0,
        a=1.0, b=1.0,
        scalar_func=overlapping_ellipse_scalar_func
    )
    regions.append(overlapping_ellipse_region)

    # Define Boundary Conditions as Functions
    boundary_conditions = {
        'left': lambda y: np.sin(np.pi * y / Ly),      # Sine wave on the left boundary
        'right': lambda y: 0.0,                        # Constant zero on the right boundary
        'top': lambda x: np.sqrt(x),       # Cosine wave on the top boundary
        'bottom': lambda x: 0.0                        # Constant zero on the bottom boundary
    }

    # Run simulation
    X, Y, U_history, dt = simulate_diffusion_2D(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        T=T,
        D=D,
        regions=regions,
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
