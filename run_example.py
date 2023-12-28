import logging
import random

import torch
from matplotlib import pyplot as plt

import cegis_cbf.cegis
from cegis_cbf.common import domains
from cegis_cbf.common.consts import ActivationType
from cegis_cbf.common.consts import CertificateType, TimeDomain, VerifierType
from cegis_cbf.common.plotting import benchmark_3d
from systems import make_system


def main():
    seed = 916104
    system_name = "single_integrator"
    n_hidden_neurons = 10
    activations = (ActivationType.RELU, ActivationType.LINEAR)
    n_data_samples = 500
    verbose = 0

    log_levels = [logging.INFO, logging.DEBUG]
    logging.basicConfig(level=log_levels[verbose])

    n_hidden_neurons = (n_hidden_neurons,) * len(activations)

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
    data_gen = {
        "lie": lambda n: torch.concatenate([XD.generate_data(n), UD.generate_data(n)], dim=1),
        "init": lambda n: XI.generate_data(n),
        "unsafe": lambda n: XU.generate_data(n),
    }

    config = cegis_cbf.cegis.CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA_GEN=data_gen,
        CERTIFICATE=CertificateType.CBF,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.Z3,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=100,
        N_DATA=n_data_samples,
        SEED=seed,
    )
    cegis = cegis_cbf.cegis.Cegis(config=config, verbose=verbose)

    result = cegis.solve()

    if XD.dimension == 2:
        plt.clf()

        xrange = (XD.lower_bounds[0], XD.upper_bounds[0])
        yrange = (XD.lower_bounds[1], XD.upper_bounds[1])

        #ax1 = benchmark_plane(closed_loop_model, [result.cert], config.DOMAINS, levels, xrange, yrange)
        ax2 = benchmark_3d(result.net, config.DOMAINS, [0.0], xrange, yrange, title="CBF")
        #ax3 = benchmark_lie(closed_loop_model, [result.cert], config.DOMAINS, levels, xrange, yrange)

        plt.savefig(f"cbf_final.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
