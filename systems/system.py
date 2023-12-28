from abc import abstractmethod

import numpy as np
import torch
import z3

from fosco.common.utils import contains_object


class ControlAffineControllableDynamicalModel:
    """
    Implements a controllable dynamical model with control-affine dynamics dx = f(x) + g(x) u
    """

    @property
    @abstractmethod
    def n_vars(self) -> int:
        raise NotImplementedError()

    @property
    @abstractmethod
    def n_controls(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def fx_torch(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def fx_smt(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gx_torch(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def gx_smt(self, x) -> np.ndarray | torch.Tensor:
        raise NotImplementedError()

    def f(
        self, v: np.ndarray | torch.Tensor, u: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return self._f_torch(v, u)
        elif contains_object(v, z3.ArithRef):
            dvs = self.fx_smt(v) + self.gx_smt(v) @ u
            return [z3.simplify(dv) for dv in dvs]
        else:
            raise NotImplementedError(f"Unsupported type {type(v)}")

    def _f_torch(self, v: torch.Tensor, u: torch.Tensor) -> list:
        v = v.reshape(-1, self.n_vars, 1)
        u = u.reshape(-1, self.n_controls, 1)
        vdot = self.fx_torch(v) + self.gx_torch(v) @ u
        return vdot.reshape(-1, self.n_vars)

    def __call__(
        self, v: np.ndarray | torch.Tensor, u: np.ndarray | torch.Tensor
    ) -> np.ndarray | torch.Tensor:
        return self.f(v, u)
