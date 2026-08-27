import torch
import torch.nn as nn
from torchdiffeq import odeint

###Create LatentODEFunc Class --> OK
class LatentODEFunc(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=256):  # Upgraded width
        super().__init__()
        self.latent_dim = latent_dim

        # A simple MLP to represent the velocity field f(z, t, goal)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + latent_dim, hidden_dim),  # Input: current_z + goal_p2
            nn.ELU(),  # Smooth, efficient activation
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim) # Output: velocity (dz/dt)
        )

    def forward(self, t, z, p2):
        # z shape might be [Batch, LatentDim] or [Batch, 1, LatentDim]
        # p2 shape is [Batch, LatentDim]
        # Check if z has an extra dimension (like Time) that p2 doesn't have
        if z.dim() > p2.dim():
            # This handles the case where z is [T, B, 2] and p2 is [B, 2]
            # We add a leading dimension to p2 and expand to match z
            # dims_to_add determines how many 1s to add to the front
            diff = z.dim() - p2.dim()
            p2_ready = p2.view(*([1] * diff), *p2.shape).expand_as(z)
        else:
            # Standard case: both are [Batch, 2]
            p2_ready = p2

        # Concatenation of the current latent position with the fixed goal
        # This makes the ODE "Goal-Conditioned"
        combined = torch.cat([z, p2_ready], dim=-1)
        velocity = self.net(combined)
        return velocity

###Create GoalConditionedNODE Class --> OK
class GoalConditionedNODE(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=256):
        super().__init__()
        # Pass the hidden_dim down to the function
        self.ode_func = LatentODEFunc(latent_dim, hidden_dim)

    def forward(self, p1, p2, t_steps):
        """
        p1: Start point [Batch, 2]
        p2: Goal point  [Batch, 2]
        t_steps: Time points to evaluate [T]
        """
        # We wrap the integration to return both positions AND velocities
        # for our Energy Loss calculation.

        # 1. Integrate positions using a solver (e.g., torchode or torchdiffeq)
        # For simplicity, here is the logical flow:
        z_traj = odeint(lambda t, z: self.ode_func(t, z, p2), p1, t_steps, method='rk4')

        # 2. Calculate velocities at those predicted positions
        # We need this for v^T G v
        # We broadcast t_steps to match z_traj if ode_func needs t for every point
        v_traj = self.ode_func(t_steps[:, None, None], z_traj, p2)

        # Combined output shape: [Time, Batch, 2 (Pos/Vel), LatentDim]
        return torch.stack([z_traj, v_traj], dim=2)
