import numpy as np
from abc import ABC, abstractmethod

class Region(ABC):
    """
    Abstract base class for defining regions with scalar functions.
    """
    @abstractmethod
    def apply(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Apply the region's scalar function to the grid.

        Parameters:
        - X, Y: 2D spatial grid points.

        Returns:
        - U_region: 2D array with scalar values for this region.
        """
        pass

class RectangularRegion(Region):
    """
    Represents a rectangular region with a scalar function.
    """
    def __init__(self, start_x: float, end_x: float, start_y: float, end_y: float, scalar_func):
        """
        Initialize a rectangular region.

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

    def apply(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        mask = (X >= self.start_x) & (X <= self.end_x) & (Y >= self.start_y) & (Y <= self.end_y)
        U_region = np.where(mask, self.scalar_func(X, Y), 0.0)
        return U_region

class EllipticRegion(Region):
    """
    Represents an elliptic region with a scalar function.
    """
    def __init__(self, x_center: float, y_center: float, a: float, b: float, scalar_func):
        """
        Initialize an elliptic region.

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

    def apply(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        ellipse_eq = ((X - self.x_center) / self.a)**2 + ((Y - self.y_center) / self.b)**2
        mask = ellipse_eq <= 1
        U_region = np.where(mask, self.scalar_func(X, Y), 0.0)
        return U_region

class InitialConditions:
    """
    Prepares the initial and boundary conditions for the diffusion simulation.
    """
    def __init__(self, Lx: float, Ly: float, Nx: int, Ny: int, regions: list, boundary_conditions: dict):
        """
        Initialize the initial conditions.

        Parameters:
        - Lx: Length of the domain in x-direction.
        - Ly: Length of the domain in y-direction.
        - Nx: Number of spatial grid points in x-direction.
        - Ny: Number of spatial grid points in y-direction.
        - regions: List of Region objects defining initial scalar distributions.
        - boundary_conditions: Dictionary with keys 'left', 'right', 'top', 'bottom' specifying boundary functions or constants.
        """
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.regions = regions
        self.boundary_conditions = boundary_conditions

        self.X, self.Y, self.dx, self.dy = self.initialize_domain()
        self.U_initial = self.initialize_field()

    def initialize_domain(self):
        """
        Initialize the 2D spatial domain.

        Returns:
        - X, Y: 2D spatial grid points.
        - dx, dy: Spatial step sizes in x and y directions.
        """
        x = np.linspace(0, self.Lx, self.Nx)
        y = np.linspace(0, self.Ly, self.Ny)
        X, Y = np.meshgrid(x, y)
        dx = self.Lx / (self.Nx - 1)
        dy = self.Ly / (self.Ny - 1)
        return X, Y, dx, dy

    def initialize_field(self):
        """
        Initialize the scalar field by applying all regions.

        Returns:
        - U: Initial scalar quantity distribution as a 2D array.
        """
        U = np.zeros_like(self.X)
        for region in self.regions:
            U += region.apply(self.X, self.Y)
        U = self.apply_boundary_conditions(U)
        return U

    def apply_boundary_conditions(self, U: np.ndarray) -> np.ndarray:
        """
        Apply boundary conditions to the scalar field.

        Parameters:
        - U: Current scalar quantity distribution as a 2D array.

        Returns:
        - U: Updated scalar quantity distribution with boundary conditions applied.
        """
        # Left Boundary (x = 0), varies with y
        if 'left' in self.boundary_conditions:
            bc_left = self.boundary_conditions['left']
            if callable(bc_left):
                U[:, 0] = bc_left(self.Y[:, 0])
            else:
                U[:, 0] = bc_left
        else:
            U[:, 0] = 0.0  # Default value if not specified

        # Right Boundary (x = Lx), varies with y
        if 'right' in self.boundary_conditions:
            bc_right = self.boundary_conditions['right']
            if callable(bc_right):
                U[:, -1] = bc_right(self.Y[:, -1])
            else:
                U[:, -1] = bc_right
        else:
            U[:, -1] = 0.0  # Default value if not specified

        # Bottom Boundary (y = 0), varies with x
        if 'bottom' in self.boundary_conditions:
            bc_bottom = self.boundary_conditions['bottom']
            if callable(bc_bottom):
                U[0, :] = bc_bottom(self.X[0, :])
            else:
                U[0, :] = bc_bottom
        else:
            U[0, :] = 0.0  # Default value if not specified

        # Top Boundary (y = Ly), varies with x
        if 'top' in self.boundary_conditions:
            bc_top = self.boundary_conditions['top']
            if callable(bc_top):
                U[-1, :] = bc_top(self.X[-1, :])
            else:
                U[-1, :] = bc_top
        else:
            U[-1, :] = 0.0  # Default value if not specified

        return U
