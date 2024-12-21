from initial_conditions import RectangularRegion, EllipticRegion, InitialConditions
from simulator import Simulator
from visualizer import Visualizer
import numpy as np

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
        'top': lambda x: np.sqrt(x),       # Square root function on the top boundary
        'bottom': lambda x: 0.0                        # Constant zero on the bottom boundary
    }

    # Initialize InitialConditions
    initial_conditions = InitialConditions(
        Lx=Lx,
        Ly=Ly,
        Nx=Nx,
        Ny=Ny,
        regions=regions,
        boundary_conditions=boundary_conditions
    )

    # Initialize Simulator
    simulator = Simulator(
        X=initial_conditions.X,
        Y=initial_conditions.Y,
        U_initial=initial_conditions.U_initial,
        boundary_conditions=boundary_conditions,
        D=D,
        T=T
    )

    # Run Simulation
    U_history, dt = simulator.run()

    # Initialize Visualizer
    visualizer = Visualizer(
        X=initial_conditions.X,
        Y=initial_conditions.Y,
        U_history=U_history,
        dt=dt
    )

    # Plot Results
    visualizer.plot()

if __name__ == "__main__":
    main()
