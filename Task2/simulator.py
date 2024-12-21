import numpy as np

class Simulator:
    """
    Runs the 2D diffusion simulation.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, U_initial: np.ndarray, boundary_conditions: dict, 
                 D: float, T: float):
        """
        Initialize the simulator.

        Parameters:
        - X, Y: 2D spatial grid points.
        - U_initial: Initial scalar quantity distribution as a 2D array.
        - boundary_conditions: Dictionary with keys 'left', 'right', 'top', 'bottom' specifying boundary functions or constants.
        - D: Diffusion coefficient.
        - T: Total simulation time.
        """
        self.X = X
        self.Y = Y
        self.U = U_initial.copy()
        self.boundary_conditions = boundary_conditions
        self.D = D
        self.T = T

        self.dx = X[0,1] - X[0,0]
        self.dy = Y[1,0] - Y[0,0]

        # Time step based on stability criterion for 2D: alpha + beta <= 0.5
        self.dt = 0.25 * min(self.dx, self.dy)**2 / self.D  # Safety factor < 0.25 for 2D stability
        self.Nt = int(self.T / self.dt)
        self.alpha = self.D * self.dt / self.dx**2
        self.beta = self.D * self.dt / self.dy**2

        print(f"Simulation Parameters:")
        print(f"Time Step (dt): {self.dt}")
        print(f"Total Time Steps (Nt): {self.Nt}")
        print(f"Alpha (D*dt/dx^2): {self.alpha}")
        print(f"Beta (D*dt/dy^2): {self.beta}")

    def diffusion_step(self) -> np.ndarray:
        """
        Perform a single time step update using the explicit FDM scheme for 2D.

        Returns:
        - U_new: Updated scalar quantity distribution after one time step.
        """
        U_new = self.U.copy()
        # Update internal points using vectorized operations
        U_new[1:-1,1:-1] = self.U[1:-1,1:-1] + \
                            self.alpha * (self.U[1:-1, 2:] - 2 * self.U[1:-1,1:-1] + self.U[1:-1, :-2]) + \
                            self.beta * (self.U[2:,1:-1] - 2 * self.U[1:-1,1:-1] + self.U[:-2,1:-1])
        return U_new

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

    def run(self) -> tuple:
        """
        Run the diffusion simulation.

        Returns:
        - U_history: History of scalar quantity distributions over time as a 3D array.
        - dt: Time step size.
        """
        U = self.U.copy()
        U = self.apply_boundary_conditions(U)
        U_history = [U.copy()]

        for n in range(1, self.Nt + 1):
            U = self.diffusion_step()
            U = self.apply_boundary_conditions(U)
            self.U = U.copy()
            U_history.append(U.copy())
            if n % max(1, (self.Nt // 10)) == 0:
                print(f"Progress: {n}/{self.Nt} steps completed.")

        U_history = np.array(U_history)
        return U_history, self.dt
