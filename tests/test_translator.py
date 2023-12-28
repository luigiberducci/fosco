import unittest

import numpy as np
import torch

from cegis_cbf.models.network import MLP
from cegis_cbf.translator import MLPZ3Translator, make_translator
from cegis_cbf.verifier import VerifierZ3


class TestTranslator(unittest.TestCase):
    def test_translator_linear_layer(self):
        import z3

        n_vars = 2

        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = MLP(input_size=n_vars, hidden_sizes=(), activation=(), output_size=1)

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPZ3Translator(rounding=-1)

        expr_nn, expr_nndot = translator.get_symbolic_formula(x, nn, xdot)
        assert isinstance(expr_nn, z3.ArithRef)
        assert isinstance(expr_nndot, z3.ArithRef)

        w1 = nn.W1.detach().numpy().flatten()
        b1 = nn.b1.detach().numpy().flatten()

        expected_expr_nn = w1 @ x + b1
        grad_nn = w1
        expected_expr_nndot = grad_nn @ xdot

        expected_expr_nn = z3.simplify(expected_expr_nn[0])
        expected_expr_nndot = z3.simplify(expected_expr_nndot[0])

        assert str(expr_nn) == str(
            expected_expr_nn
        ), f"Wrong symbolic formula for V, got {expr_nn}"
        assert str(expr_nndot) == str(
            expected_expr_nndot
        ), f"Wrong symbolic formula for Vdot, got {expr_nndot}"

    def test_separation_symbolic_functions(self):
        n_vars = 2

        x = VerifierZ3.new_vars(n_vars, base="x")
        x = np.array(x).reshape(-1, 1)

        nn = MLP(input_size=n_vars, hidden_sizes=(), activation=(), output_size=1)

        xdot = np.array(x).reshape(-1, 1)

        translator = MLPZ3Translator(rounding=-1)

        expr_nn, expr_nndot = translator.get_symbolic_formula(x, nn, xdot)

        expr_nn2 = translator.get_symbolic_net(x, nn)
        expr_nn_grad = translator.get_symbolic_net_grad(x, nn)
        expr_nndot2 = (expr_nn_grad @ xdot)[0, 0]

        assert str(expr_nn) == str(
            expr_nn2
        ), f"Wrong symbolic formula for V, got {expr_nn}"
        assert str(expr_nndot) == str(
            expr_nndot2
        ), f"Wrong symbolic formula for Vdot, got {expr_nndot}"

    def test_factory(self):
        from cegis_cbf.common.consts import VerifierType
        from cegis_cbf.common.consts import TimeDomain

        translator = make_translator(
            verifier_type=VerifierType.Z3, time_domain=TimeDomain.CONTINUOUS
        )
        self.assertTrue(isinstance(translator, MLPZ3Translator))

        self.assertRaises(
            NotImplementedError,
            make_translator,
            verifier_type=VerifierType.Z3,
            time_domain=TimeDomain.DISCRETE,
        )
