import unittest

import numpy as np
import torch
import z3




class TestControlAffineDynamicalSystem(unittest.TestCase):

    def test_single_integrator(self):
        from systems.single_integrator import SingleIntegrator

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        T = 10.0
        dt = 0.1

        f = SingleIntegrator()

        t = dt
        while t < T:
            x = x + dt * f(x, u)
            t += dt

        self.assertTrue(np.allclose(x, 10.0 * np.ones_like(x)), f"got {x}")

    def test_single_integrator_z3(self):
        from systems.single_integrator import SingleIntegrator

        state_vars = ["x", "y"]
        input_vars = ["vx", "vy"]
        x = [z3.Real(var) for var in state_vars]
        u = [z3.Real(var) for var in input_vars]

        f = SingleIntegrator()

        xdot = f.f(x, u)

        self.assertTrue(str(xdot[0]) == input_vars[0], "expected xdot = vx, got {xdot[0]}")
        self.assertTrue(str(xdot[1]) == input_vars[1], "expected ydot = vy, got {xdot[1]}")





