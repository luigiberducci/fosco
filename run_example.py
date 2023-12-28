import random

import numpy as np
import torch
from matplotlib import pyplot as plt

import cegis.cegis
from cegis.common import domains
from cegis.common.activations import ActivationType
from cegis.common.consts import CertificateType, TimeDomain, VerifierType
from systems import make_system


def main():
    seed = 916104
    system_name = "single_integrator"
    n_hidden_neurons = 10
    activations = [ActivationType.RELU, ActivationType.LINEAR]

    n_hidden_neurons = [n_hidden_neurons] * len(activations)

    system = make_system(id=system_name)
    if system_name == "single_integrator":
        XD = domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        UD = domains.Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(-4.0, -4.0))
        XU = domains.Sphere(vars=["x0", "x1"], centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1])
    elif system_name == "double_integrator":
        XD = domains.Rectangle(vars=["x0", "x1", "x2", "x3"],
                                      lb=(-5.0, -5.0, -5.0, -5.0),
                                      ub=(5.0, 5.0, 5.0, 5.0))
        UD = domains.Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = domains.Rectangle(vars=["x0", "x1", "x2", "x3"],
                                      lb=(-5.0, -5.0, -5.0, -5.0),
                                      ub=(-4.0, -4.0, 5.0, 5.0))
        XU = domains.Rectangle(vars=["x0", "x1", "x2", "x3"],
                                      lb=(-1.0, -1.0, -5.0, -5.0),
                                      ub=(1.0, 1.0, 5.0, 5.0))
    else:
        raise NotImplementedError(f"System {system_name} not implemented")

    # seeding
    if seed is None:
        seed = random.randint(0, 1000000)
    print("Seed:", seed)

    sets = {
        "lie": XD,
        "input": UD,
        "init": XI,
        "unsafe": XU,
    }
    data = {
        "lie": lambda n: torch.concatenate([XD.generate_data(n), UD.generate_data(n)], dim=1),
        "init": lambda n: XI._generate_data(n),
        "unsafe": lambda n: XU._generate_data(n),
    }

    opts = cegis.CegisConfig(
        N_VARS=system().n_vars,
        N_CONTROLS=system().n_controls,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.CBF,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.Z3,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=100,
        VERBOSE=1,
        SEED=seed,
    )

    levels = [[0.0]]

    result = fossil.synthesise(
        opts,
    )

    ctrl = control.DummyController(
        inputs=opts.N_VARS,
        output=opts.N_CONTROLS,
        const_out=1.0
    )
    closed_loop_model = control.GeneralClosedLoopModel(result.f, ctrl)

    if XD.dimension == 2:
        xrange = (XD.lower_bounds[0], XD.upper_bounds[0])
        yrange = (XD.lower_bounds[1], XD.upper_bounds[1])

        ax1 = benchmark_plane(closed_loop_model, [result.cert], opts.DOMAINS, levels, xrange, yrange)
        ax2 = benchmark_3d([result.cert], opts.DOMAINS, levels, xrange, yrange)
        ax3 = benchmark_lie(closed_loop_model, [result.cert], opts.DOMAINS, levels, xrange, yrange)

        plt.show()


if __name__ == "__main__":
    main()
