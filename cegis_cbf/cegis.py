import logging
import math
from collections import namedtuple
from dataclasses import dataclass, replace
from typing import Any, Type

import matplotlib.pyplot as plt
import torch

from cegis_cbf.cbf import ControlBarrierFunction
from cegis_cbf.common.formatter import CustomFormatter
from cegis_cbf.common.plotting import benchmark_3d
from cegis_cbf.consolidator import Consolidator, make_consolidator
from cegis_cbf.common.consts import CertificateType, TimeDomain, ActivationType, VerifierType, DomainNames
from cegis_cbf.learner import make_learner, LearnerNN
from cegis_cbf.translator import make_translator
from cegis_cbf.verifier import make_verifier
from systems import ControlAffineControllableDynamicalModel

CegisResult = namedtuple("CegisResult", ["found", "net", "infos"])


@dataclass
class CegisConfig:
    # system
    SYSTEM: Type[ControlAffineControllableDynamicalModel] = None
    DOMAINS: dict[str, Any] = None
    TIME_DOMAIN: TimeDomain = TimeDomain.CONTINUOUS
    # cegis_cbf
    CERTIFICATE: CertificateType = CertificateType.CBF
    VERIFIER: VerifierType = VerifierType.Z3
    CEGIS_MAX_ITERS: int = 10
    ROUNDING: int = 3
    # training
    DATA_GEN: dict[str, callable] = None
    N_DATA: int = 500
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    # net architecture
    N_HIDDEN_NEURONS: tuple[int] = (10,)
    ACTIVATION: tuple[ActivationType, ...] = (ActivationType.SQUARE,)
    # seeding
    SEED: int = None

    def __getitem__(self, item):
        return getattr(self, item)


class Cegis:
    def __init__(self, config: CegisConfig, verbose: int = 0):
        self.config = config

        # logging
        self.logger = self._initialise_logger(verbose=verbose)

        # intialization
        self.f = self.config.SYSTEM()
        self.x, self.x_map, self.domains = self._initialise_domains()
        self.xdot = self.f(**self.x_map)
        self.datasets = self._initialise_data()

        self.certificate = self._initialise_certificate()
        self.learner = self._initialise_learner()
        self.verifier = self._initialise_verifier()
        self.consolidator = self._initialise_consolidator()
        self.translator = self._initialise_translator()

        self._result = None

        self._assert_state()

    def _initialise_logger(self, verbose: int) -> logging.Logger:
        logger = logging.getLogger("CEGIS")
        ch = logging.StreamHandler()

        if verbose > 0:
            verbosity_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
            logger.setLevel(verbosity_levels[verbose])
            ch.setLevel(verbosity_levels[verbose])

        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)

        return logger



    def _initialise_learner(self) -> LearnerNN:
        learner_type = make_learner(self.config.TIME_DOMAIN)
        learner_instance = learner_type(
            input_size=self.f.n_vars,
            learn_method=self.certificate.learn,
            hidden_sizes=self.config.N_HIDDEN_NEURONS,
            activation=self.config.ACTIVATION,
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )
        return learner_instance

    def _initialise_verifier(self):
        verifier_type = make_verifier(self.config.VERIFIER)
        verifier_instance = verifier_type(
            solver_vars=self.x,
            constraints_method=self.certificate.get_constraints
        )
        return verifier_instance

    def _initialise_domains(self):
        verifier_type = make_verifier(type=self.config.VERIFIER)
        x = verifier_type.new_vars(self.f.n_vars, base="x")
        u = verifier_type.new_vars(self.f.n_controls, base="u")

        # create map id -> variable
        x_map = {"v": x, "u": u}
        x = x + u

        # create domains
        domains = {
            label: domain.generate_domain(x) for label, domain in self.config.DOMAINS.items()
        }

        self.logger.debug("Domains: {}".format(domains))
        return x, x_map, domains

    def _initialise_data(self):
        datasets = {}
        for label in self.config.DATA_GEN.keys():
            datasets[label] = self.config.DATA_GEN[label](self.config.N_DATA)
        return datasets

    def _initialise_certificate(self):
        certificate_type = ControlBarrierFunction
        return certificate_type(vars=self.x, domains=self.config.DOMAINS)


    def _initialise_consolidator(self):
        return make_consolidator()

    def _initialise_translator(self):
        return make_translator(
            verifier_type=self.config.VERIFIER,
            time_domain=self.config.TIME_DOMAIN,
            rounding=self.config.ROUNDING,
        )

    def solve(self) -> CegisResult:
        state = self.init_state()

        iter = None
        for iter in range(1, self.config.CEGIS_MAX_ITERS + 1):
            self.logger.debug(f"Iteration {iter}")

            # debug print
            domains = self.config.DOMAINS
            xrange = domains["lie"].lower_bounds[0], domains["lie"].upper_bounds[0]
            yrange = domains["lie"].lower_bounds[1], domains["lie"].upper_bounds[1]
            ax2 = benchmark_3d(self.learner, domains, [0.0], xrange, yrange, title=f"CBF - Iter {iter}")
            plt.show()

            # Learner component
            self.logger.debug("Learner")
            outputs = self.learner.update(**state)
            state.update(outputs)

            # Translator component
            self.logger.debug("Translator")
            outputs = self.translator.translate(**state)
            state.update(outputs)

            # Verifier component
            self.logger.debug("Verifier")
            outputs = self.verifier.verify(**state)
            state.update(outputs)

            # Consolidator component
            self.logger.debug("Consolidator")
            outputs = self.consolidator.get(**state)
            state.update(outputs)

            if state["found"]:
                self.logger.debug("found valid certificate")
                break

        # state = self.process_timers(state)

        infos = {"iter": iter}
        self._result = CegisResult(found=state["found"], net=state["V_net"], infos=infos)

        return self._result

    def init_state(self) -> dict:
        state = {
            "found": False,
            "iter": 0,
            "system": self.f,
            "V_net": self.learner.net,
            "xdot_func": self.f._f_torch,
            "datasets": self.datasets,
            "x_v_map": self.x_map,
            "V_symbolic": None,
            "Vdot_symbolic": None,
            "xdot": self.xdot,
            "cex": None,

            # CegisStateKeys.found: False,
            # CegisStateKeys.verification_timed_out: False,
            # CegisStateKeys.cex: None,
            # CegisStateKeys.trajectory: None,
            # CegisStateKeys.ENet: self.config.ENET,

        }

        return state

    @property
    def result(self):
        return self._result

    def _assert_state(self):
        assert self.config.LEARNING_RATE > 0
        assert self.config.CEGIS_MAX_ITERS > 0
        assert self.x is self.verifier.xs, "expected same variables in cegis_cbf and verifier"
        self.certificate._assert_state(self.domains, self.datasets)
