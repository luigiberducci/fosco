import logging
import math
from typing import Generator

import torch
from torch.optim import Optimizer

from fosco.common.domains import Set, Rectangle
from fosco.common.consts import DomainNames
from fosco.common.utils import _set_assertion
from fosco.learner import LearnerNN, LearnerCT
from fosco.verifier import SYMBOL
from systems import ControlAffineControllableDynamicalModel

XD = DomainNames.XD.value
XI = DomainNames.XI.value
XU = DomainNames.XU.value
UD = DomainNames.UD.value


class ControlBarrierFunction:
    """
    Certifies Safety for continuous time controlled systems with control affine dynamics.

    Note: CBF use different conventions.
    B(Xi)>0, B(Xu)<0, Bdot(Xd) > -alpha(B(Xd)) for alpha class-k function

    Arguments:
        domains {dict}: dictionary of (string,domain) pairs
        config {CegisConfig}: configuration dictionary
    """

    def __init__(self, vars: list, domains: dict[str, Set]) -> None:
        self.x_vars = [
            v for v in vars if str(v).startswith("x")
        ]  # todo: dont like checking initial letter
        self.u_vars = [v for v in vars if str(v).startswith("u")]

        self.x_domain: SYMBOL = domains[XD].generate_domain(self.x_vars)
        self.u_set: Rectangle = domains[UD]
        self.u_domain: SYMBOL = domains[UD].generate_domain(self.u_vars)
        self.initial_domain: SYMBOL = domains[XI].generate_domain(self.x_vars)
        self.unsafe_domain: SYMBOL = domains[XU].generate_domain(self.x_vars)

        assert isinstance(
            self.u_set, Rectangle
        ), f"CBF only works with rectangular input domains, got {self.u_set}"
        self.n_vars = len(self.x_vars)
        self.n_controls = len(self.u_vars)

        # loss parameters
        # todo: bring it outside
        self.loss_relu = torch.nn.Softplus()  # torch.relu  # torch.nn.Softplus()
        self.margin = 0.0
        self.epochs = 1000

    def compute_loss(
        self,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
        alpha: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict]:
        """Computes loss function for CBF and its accuracy w.r.t. the batch of data.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            Bdot_d (torch.Tensor): Barrier derivative values for domain
            alpha (torch.Tensor): coeff. linear class-k function, f(x) = alpha * x, for alpha in R_+

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        assert (
            Bdot_d is None or B_d.shape == Bdot_d.shape
        ), f"B_d and Bdot_d must have the same shape, got {B_d.shape} and {Bdot_d.shape}"
        margin = self.margin

        accuracy_i = (B_i >= margin).count_nonzero().item()
        accuracy_u = (B_u < -margin).count_nonzero().item()
        if Bdot_d is None:
            accuracy_d = 0
            percent_accuracy_belt = 0
        else:
            accuracy_d = (Bdot_d + alpha * B_d >= margin).count_nonzero().item()
            percent_accuracy_belt = 100 * accuracy_d / Bdot_d.shape[0]
        percent_accuracy_init = 100 * accuracy_i / B_i.shape[0]
        percent_accuracy_unsafe = 100 * accuracy_u / B_u.shape[0]

        relu = self.loss_relu
        init_loss = (relu(margin - B_i)).mean()  # penalize B_i < 0
        unsafe_loss = (relu(B_u + margin)).mean()  # penalize B_u > 0
        if Bdot_d is None:
            lie_loss = 0.0
        else:
            lie_loss = (
                relu(margin - (Bdot_d + alpha * B_d))
            ).mean()  # penalize dB_d + alpha * B_d < 0

        loss = init_loss + unsafe_loss + lie_loss

        accuracy = {
            "acc init": percent_accuracy_init,
            "acc unsafe": percent_accuracy_unsafe,
            "acc derivative": percent_accuracy_belt,
        }

        # debug
        # print("\n".join([f"{k}:{v}" for k, v in accuracy.items()]))

        return loss, accuracy

    def learn(
        self,
        learner: LearnerCT,
        optimizer: Optimizer,
        datasets: dict,
        f_torch: callable,
    ) -> dict:
        """
        Updates the CBF model.

        :param learner: LearnerNN object
        :param optimizer: torch optimizer
        :param datasets: dictionary of (string,torch.Tensor) pairs
        :param f_torch: callable
        """

        condition_old = False
        i1 = datasets[XD].shape[0]
        i2 = datasets[XI].shape[0]
        # samples = torch.cat([s for s in S.values()])
        label_order = [XD, XI, XU]
        state_samples = torch.cat(
            [datasets[label][:, : self.n_vars] for label in label_order]
        )
        U_d = datasets[XD][:, self.n_vars : self.n_vars + self.n_controls]

        for t in range(self.epochs):
            optimizer.zero_grad()

            # net gradient
            B, gradB = learner.compute_net_gradnet(state_samples)

            B_d = B[:i1, 0]
            B_i = B[i1 : i1 + i2, 0]
            B_u = B[i1 + i2 :, 0]

            # compute lie derivative
            assert (
                B_d.shape[0] == U_d.shape[0]
            ), f"expected pairs of state,input data. Got {B_d.shape[0]} and {U_d.shape[0]}"
            X_d = state_samples[:i1]
            gradB_d = gradB[:i1]
            Sdot_d = f_torch(X_d, U_d)
            Bdot_d = torch.sum(torch.mul(gradB_d, Sdot_d), dim=1)

            loss, accuracy = self.compute_loss(B_i, B_u, B_d, Bdot_d, alpha=1.0)

            if t % math.ceil(self.epochs / 10) == 0 or self.epochs - t < 10:
                # log_loss_acc(t, loss, accuracy, learner.verbose)
                logging.debug(f"Epoch {t}: loss={loss}, accuracy={accuracy}")

            # early stopping after 2 consecutive epochs with ~100% accuracy
            condition = all(acc >= 99.9 for name, acc in accuracy.items())
            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizer.step()

        return {}

    def get_constraints(self, verifier, B, Bdot) -> Generator:
        """
        :param verifier: verifier object
        :param B: symbolic formula of the CBF
        :param Bdot: symbolic formula of the CBF derivative (not yet Lie derivative)
        :return: tuple of dictionaries of Barrier conditons
        """
        _True = verifier.solver_fncts()["True"]
        _And = verifier.solver_fncts()["And"]
        _Or = verifier.solver_fncts()["Or"]
        _Not = verifier.solver_fncts()["Not"]
        _Exists = verifier.solver_fncts()["Exists"]
        _ForAll = verifier.solver_fncts()["ForAll"]

        alpha = lambda x: 1.0 * x

        # find cex requires ForAll quantifier on entire input domain
        # spec := exists u Bdot + alpha * Bx >= 0 if x \in domain
        # counterexample: x s.t. forall u Bdot + alpha * Bx < 0
        #
        # smart way: verify Lie condition only on vertices of convex input space
        u_vertices = self.u_set.get_vertices()
        lie_constr = _True
        for u_vert in u_vertices:
            vertex_constr = Bdot + alpha(B) < 0
            vertex_assignment = _And(
                [u_var == u_val for u_var, u_val in zip(self.u_vars, u_vert)]
            )
            lie_constr_uv = _And(vertex_constr, vertex_assignment)
            lie_constr = _And(lie_constr, lie_constr_uv)

        # Bx >= 0 if x \in initial
        # counterexample: B < 0 and x \in initial
        initial_constr = _And(B < 0, self.initial_domain)

        # Bx < 0 if x \in unsafe
        # counterexample: B >= 0 and x \in unsafe
        unsafe_constr = _And(B >= 0, self.unsafe_domain)

        # add domain constraints
        lie_constr = _And(lie_constr, self.x_domain)
        inital_constr = _And(initial_constr, self.x_domain)
        unsafe_constr = _And(unsafe_constr, self.x_domain)

        logging.debug(f"lie_constr: {lie_constr}")
        logging.debug(f"inital_constr: {inital_constr}")
        logging.debug(f"unsafe_constr: {unsafe_constr}")

        for cs in (
            {XI: (inital_constr, self.x_vars), XU: (unsafe_constr, self.x_vars)},
            {XD: (lie_constr, self.x_vars + self.u_vars)},
        ):
            yield cs

    @staticmethod
    def _assert_state(domains, data):
        dn = DomainNames
        domain_labels = set(domains.keys())
        data_labels = set(data.keys())
        _set_assertion(
            {dn.XD.value, dn.UD.value, dn.XI.value, dn.XU.value},
            domain_labels,
            "Symbolic Domains",
        )
        _set_assertion(
            {dn.XD.value, dn.XI.value, dn.XU.value}, data_labels, "Data Sets"
        )
