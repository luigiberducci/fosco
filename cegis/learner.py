from abc import abstractmethod
from typing import Callable

import numpy as np
import torch
from torch import nn

from cegis.common.activations import activation
from cegis.common.consts import ActivationType, TimeDomain
from cegis.models.network import MLP


class LearnerNN(nn.Module):

    @abstractmethod
    def update(
            self, **kwargs
    ) -> dict:
        raise NotImplementedError

    def learn(
            self,
            datasets: torch.Tensor,
            xdot_func: Callable,
    ) -> dict:
        return self.learn_method(self, self.optimizer, datasets, xdot_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # todo: move to net?
        """Computes the value of the learner."""
        return self.net(x)

    def freeze(self):
        # todo: move to net?
        """Freezes the parameters of the neural network by setting requires_grad to False."""

        for param in self.parameters():
            if not param.requires_grad:
                break
            param.requires_grad = False

    def compute_V_gradV(
        self, nn: torch.Tensor, grad_nn: torch.Tensor, S: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the value of the function and its gradient.

        The function is defined as:
            V = NN(x) * F(x)
        where NN(x) is the neural network and F(x) is a factor, equal to either 1 or ||x||^2.

            nn (torch.Tensor): neural network value
            grad_nn (torch.Tensor): gradient of the neural network
            S (torch.Tensor): input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (V, gradV)
        """
        F, derivative_F = self.compute_factors(S)
        V = nn * F
        # define F(x) := ||x||^2
        # V = NN(x) * F(x)
        # gradV = NN(x) * dF(x)/dx  + der(NN) * F(x)
        # gradV = torch.stack([nn, nn]).T * derivative_e + grad_nn * torch.stack([E, E]).T
        if self.factor is not None:
            gradV = (
                nn.expand_as(grad_nn.T).T * derivative_F.expand_as(grad_nn)
                + grad_nn * F.expand_as(grad_nn.T).T
            )
        else:
            gradV = grad_nn
        return V, gradV


class LearnerCT(LearnerNN):
    """Leaner class for continuous time dynamical models.

    Learns and evaluates V and Vdot.

    """

    def __init__(
            self,
            input_size,
            learn_method,
            hidden_sizes: tuple[int, ...],
            activation: tuple[ActivationType, ...],
            lr: float,
            weight_decay: float,
    ):
        super(LearnerCT, self).__init__()

        self.net = MLP(
            input_size=input_size,
            output_size=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        )

        self.optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.learn_method = learn_method

    def update(
            self, datasets, xdot_func, **kwargs
    ) -> dict:
        output = self.learn(datasets=datasets, xdot_func=xdot_func)
        return output

    def compute_net_gradnet(self, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the value of the neural network and its gradient.

        Computes gradient using autograd.

            S (torch.Tensor): input tensor

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (nn, grad_nn)
        """
        S_clone = torch.clone(S).requires_grad_()
        nn = self(S_clone)

        grad_nn = torch.autograd.grad(
            outputs=nn,
            inputs=S_clone,
            grad_outputs=torch.ones_like(nn),
            create_graph=True,
            retain_graph=True,
        )[0]
        return nn, grad_nn

    def compute_dV(self, gradV: torch.Tensor, Sdot: torch.Tensor) -> torch.Tensor:
        """Computes the  lie derivative of the function.

        Args:
            gradV (torch.Tensor): gradient of the function
            Sdot (torch.Tensor): df/dt

        Returns:
            torch.Tensor: dV/dt
        """
        # Vdot = gradV * f(x)
        Vdot = torch.sum(torch.mul(gradV, Sdot), dim=1)
        return Vdot


def make_learner(time_domain: TimeDomain) -> Callable:
    if time_domain == TimeDomain.CONTINUOUS:
        return LearnerCT
    else:
        raise NotImplementedError(f"Unsupported time domain {time_domain}")
