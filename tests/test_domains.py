import unittest

from fosco.common import domains


class TestDomains(unittest.TestCase):
    def test_rectangle(self):
        X = domains.Rectangle(vars=["x", "y"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        self.assertEqual(X.dimension, 2)
        self.assertEqual(X.vars, ["x", "y"])
        self.assertEqual(X.lower_bounds, (-5.0, -5.0))
        self.assertEqual(X.upper_bounds, (5.0, 5.0))

        data = X.generate_data(1000)
        self.assertEqual(data.shape, (1000, 2))

        for sample in data:
            x, y = sample
            self.assertGreaterEqual(x, -5.0)
            self.assertLessEqual(x, 5.0)
            self.assertGreaterEqual(y, -5.0)
            self.assertLessEqual(y, 5.0)

    def test_sphere(self):
        X = domains.Sphere(vars=["x", "y", "z"], centre=(0.0, 0.0, 0.0), radius=5.0)
        self.assertEqual(X.dimension, 3)

        data = X.generate_data(1000)
        self.assertEqual(data.shape, (1000, 3))

        for sample in data:
            x, y, z = sample
            self.assertLessEqual(x ** 2 + y ** 2 + z ** 2, 5.0 ** 2)
