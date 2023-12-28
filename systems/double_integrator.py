import numpy as np
import torch

from systems import ControlAffineControllableDynamicalModel


class DoubleIntegrator(ControlAffineControllableDynamicalModel):
    """
    Single integrator system. X=[x, y], U=[vx, vy]
    dX/dt = [vx, vy]
    """

    @property
    def n_vars(self) -> int:
        return 4

    @property
    def n_controls(self) -> int:
        return 2

    def fx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
            len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            vx, vy = x[:, 2, :], x[:, 3, :]
            fx = np.concatenate([vx, vy, np.zeros_like(vx), np.zeros_like(vy)], axis=1)
            fx = fx[:, :, None]
        else:
            vx, vy = x[:, 2, :], x[:, 3, :]
            fx = torch.cat([vx, vy, torch.zeros_like(vx), torch.zeros_like(vy)], dim=1)
            fx = fx[:, :, None]
        return fx

    def fx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        vx, vy = x[2], x[3]
        return np.array([vx, vy, 0, 0])

    def gx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert (
            len(x.shape) == 3
        ), "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            gx = np.zeros((self.n_vars, self.n_controls))
            gx[2:, :] = np.eye(self.n_controls)
            gx = gx[None].repeat(x.shape[0], axis=0)
        else:
            gx = torch.zeros((self.n_vars, self.n_controls))
            gx[2:, :] = torch.eye(self.n_controls)
            gx = gx[None].repeat((x.shape[0], 1, 1))
        return gx

    def gx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(
            x, list
        ), "expected list of symbolic state variables, [x0, x1, ...]"
        gx = np.zeros((self.n_vars, self.n_controls))
        gx[2:, :] = np.eye(self.n_controls)
        return gx
