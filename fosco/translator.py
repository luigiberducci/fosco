from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import z3

from fosco.common.activations_symbolic import activation_sym, activation_der_sym
from fosco.common.consts import VerifierType, TimeDomain
from fosco.models.network import MLP
from fosco.verifier import SYMBOL


class Translator(ABC):
    """
    Abstract class for symbolic translators.
    """

    @abstractmethod
    def translate(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_symbolic_net(self, input_vars: Iterable[SYMBOL], net: MLP) -> SYMBOL:
        """
        Translate a network forward pass into a symbolic expression.

        :param input_vars: list of symbolic variables
        :param net: network model
        :return: symbolic expression
        """
        raise NotImplementedError

    @abstractmethod
    def get_symbolic_net_grad(
        self, input_vars: Iterable[SYMBOL], net: MLP
    ) -> Iterable[SYMBOL]:
        """
        Translate the network gradient w.r.t. the input into a symbolic expression.

        :param input_vars: list of symbolic variables
        :param net: network model
        :return:
        """
        raise NotImplementedError


class MLPZ3Translator(Translator):
    """
    Symbolic translator for feed-forward neural networks to z3 expressions.
    """

    def __init__(self, rounding: int = 3):
        self.round = rounding

    def translate(
        self,
        x_v_map: dict[str, Iterable[SYMBOL]],
        V_net: MLP,
        xdot: Iterable[SYMBOL],
        **kwargs,
    ):
        x_vars = x_v_map["v"]
        xdot = np.array(xdot).reshape(-1, 1)
        V_symbolic, Vdot_symbolic = self.get_symbolic_formula(x_vars, V_net, xdot)

        assert isinstance(
            V_symbolic, z3.ArithRef
        ), f"Expected V_symbolic to be z3.ArithRef, got {type(V_symbolic)}"
        assert isinstance(
            Vdot_symbolic, z3.ArithRef
        ), f"Expected Vdot_symbolic to be z3.ArithRef, got {type(Vdot_symbolic)}"

        return {
            "V_symbolic": V_symbolic,
            "Vdot_symbolic": Vdot_symbolic,
        }

    def get_symbolic_net(self, input_vars: Iterable[SYMBOL], net: MLP) -> SYMBOL:
        """
        Translate a MLP forward pass into a symbolic expression.

        :param input_vars: list of symbolic variables
        :param net: network model
        :return: symbolic expression
        """
        input_vars = np.array(input_vars).reshape(-1, 1)

        z, _ = self.network_until_last_layer(net, input_vars)

        if self.round < 0:
            last_layer = net.layers[-1].weight.data.numpy()
        else:
            last_layer = np.round(net.layers[-1].weight.data.numpy(), self.round)

        z = last_layer @ z
        if net.layers[-1].bias is not None:
            z += net.layers[-1].bias.data.numpy()[:, None]
        assert z.shape == (1, 1), f"Wrong shape of z, expected (1, 1), got {z.shape}"

        V = z[0, 0]
        V = z3.simplify(V)

        return V

    def get_symbolic_net_grad(
        self, input_vars: Iterable[SYMBOL], net: MLP
    ) -> Iterable[SYMBOL]:
        """
        Translate the MLP gradient w.r.t. the input into a symbolic expression.

        :param input_vars: list of symbolic variables
        :param net: network model
        :return:
        """
        input_vars = np.array(input_vars).reshape(-1, 1)

        z, jacobian = self.network_until_last_layer(net, input_vars)

        if self.round < 0:
            last_layer = net.layers[-1].weight.data.numpy()
        else:
            last_layer = np.round(net.layers[-1].weight.data.numpy(), self.round)

        z = last_layer @ z
        if net.layers[-1].bias is not None:
            z += net.layers[-1].bias.data.numpy()[:, None]
        gradV = last_layer @ jacobian  # jacobian now contains the grad V

        assert z.shape == (1, 1)
        assert gradV.shape == (
            1,
            net.input_size,
        ), f"Wrong shape of gradV, expected (1, {net.input_size}), got {gradV.shape}"

        # z3 simplification
        for i in range(net.input_size):
            gradV[0, i] = (
                z3.simplify(gradV[0, i])
                if isinstance(gradV[0, i], z3.ArithRef)
                else gradV[0, i]
            )

        return gradV

    def get_symbolic_formula(self, x, net, xdot):
        """
        Original implementation for continuous time models, keeping for reference.

        :param net:
        :param x:
        :param xdot:
        :return:
        """
        x = np.array(x).reshape(-1, 1)

        z, jacobian = self.network_until_last_layer(net, x)

        if self.round < 0:
            last_layer = net.layers[-1].weight.data.numpy()
        else:
            last_layer = np.round(net.layers[-1].weight.data.numpy(), self.round)

        z = last_layer @ z
        if net.layers[-1].bias is not None:
            z += net.layers[-1].bias.data.numpy()[:, None]
        gradV = last_layer @ jacobian  # jacobian now contains the grad V

        assert z.shape == (1, 1)
        assert gradV.shape == (
            1,
            net.input_size,
        ), f"Wrong shape of gradV, expected (1, {net.input_size}), got {gradV.shape}"

        Vdot = gradV @ xdot

        V = z[0, 0]
        Vdot = Vdot[0, 0]

        # z3 simplification
        V = z3.simplify(V)
        Vdot = z3.simplify(Vdot)

        return V, Vdot

    def network_until_last_layer(
        self, net: MLP, input_vars: Iterable[SYMBOL]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Symbolic forward pass excluding the last layer.

        :param net: network model
        :param input_vars: list of symbolic variables
        :return: tuple (net output, its jacobian)
        """
        z = input_vars
        jacobian = np.eye(net.input_size, net.input_size)

        for idx, layer in enumerate(net.layers[:-1]):
            if self.round < 0:
                w = layer.weight.data.numpy()
                if layer.bias is not None:
                    b = layer.bias.data.numpy()[:, None]
                else:
                    b = np.zeros((layer.out_features, 1))
            elif self.round >= 0:
                w = np.round(layer.weight.data.numpy(), self.round)
                if layer.bias is not None:
                    b = np.round(layer.bias.data.numpy(), self.round)[:, None]
                else:
                    b = np.zeros((layer.out_features, 1))

            zhat = w @ z + b
            z = activation_sym(net.acts[idx], zhat)

            jacobian = w @ jacobian
            jacobian = np.diagflat(activation_der_sym(net.acts[idx], zhat)) @ jacobian

        return z, jacobian


def make_translator(
    verifier_type: VerifierType, time_domain: TimeDomain, **kwargs
) -> Translator:
    """
    Factory function for translators.
    """
    if verifier_type == VerifierType.Z3 and time_domain == TimeDomain.CONTINUOUS:
        return MLPZ3Translator(**kwargs)
    else:
        raise NotImplementedError(
            f"Translator for verifier={verifier_type} and time={time_domain} not implemented"
        )
