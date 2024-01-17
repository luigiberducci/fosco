import logging
import unittest

import torch

from fosco.cegis import CegisConfig, Cegis
from fosco.common.domains import Rectangle, Sphere
from fosco.common.consts import (
    TimeDomain,
    ActivationType,
    VerifierType,
    CertificateType,
    DomainNames,
)
from systems.single_integrator import SingleIntegrator


class TestCEGIS(unittest.TestCase):
    def _get_single_integrator_config(self):
        XD = Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        UD = Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(-4.0, -4.0))
        XU = Sphere(vars=["x0", "x1"], centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1])

        dn = DomainNames
        domains = {
            name: domain
            for name, domain in zip(
                [dn.XD.value, dn.UD.value, dn.XI.value, dn.XU.value], [XD, UD, XI, XU]
            )
        }

        data_gen = {
            dn.XD.value: lambda n: torch.concatenate(
                [XD.generate_data(n), UD.generate_data(n)], dim=1
            ),
            dn.XI.value: lambda n: XI.generate_data(n),
            dn.XU.value: lambda n: XU.generate_data(n),
        }

        config = CegisConfig(
            SYSTEM=SingleIntegrator,
            DOMAINS=domains,
            TIME_DOMAIN=TimeDomain.CONTINUOUS,
            CERTIFICATE=CertificateType.CBF,
            VERIFIER=VerifierType.Z3,
            CEGIS_MAX_ITERS=5,
            ROUNDING=3,
            DATA_GEN=data_gen,
            N_DATA=500,
            LEARNING_RATE=1e-3,
            WEIGHT_DECAY=1e-4,
            N_HIDDEN_NEURONS=(10, 10,),
            ACTIVATION=(ActivationType.RELU, ActivationType.LINEAR),
            SEED=0,
        )

        return config

    def test_loop(self):
        config = self._get_single_integrator_config()
        config.LEARNING_RATE = 1e-30  # make learning rate small so that we don't learn anything in 10 iters

        c = Cegis(config=config, verbose=2)
        results = c.solve()

        infos = results.infos
        self.assertTrue(
            infos["iter"] == config.CEGIS_MAX_ITERS,
            f"Did not run for {config.CEGIS_MAX_ITERS} iterations, iter={infos['iter']}",
        )
