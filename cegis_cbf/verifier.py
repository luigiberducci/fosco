import logging
import timeit
from typing import Callable, Generator, Iterable, Type

import torch
import z3

from cegis_cbf.common.utils import contains_object
from cegis_cbf.common.consts import VerifierType

SYMBOL = z3.ArithRef
INF: float = 1e300

FUNCTIONS = {
    "And": z3.And,
    "Or": z3.Or,
    "If": z3.If,
    "Not": z3.Not,
    "False": False,
    "True": True,
    "Exists": z3.Exists,
    "ForAll": z3.ForAll,
    "Check": lambda x: contains_object(x, z3.ArithRef),
}


class Verifier:
    def __init__(
        self, constraints_method: Callable[[], Generator], solver_vars: Iterable[SYMBOL]
    ):
        super().__init__()
        self.xs = solver_vars
        self.n = len(solver_vars)
        self.constraints_method = constraints_method

        # internal vars
        self.iter = -1
        self._last_cex = []

        # params, todo: make them configurable
        self.counterexample_n = 20
        self._n_cex_to_keep = self.counterexample_n * 1
        self._solver_timeout = 30

        assert self.counterexample_n > 0
        assert self._n_cex_to_keep > 0
        assert self._solver_timeout > 0

    @staticmethod
    def new_vars(n, base: str = "x") -> list[SYMBOL]:
        raise NotImplementedError("")

    @staticmethod
    def solver_fncts():
        raise NotImplementedError("")

    @staticmethod
    def new_solver(self):
        raise NotImplementedError("")

    @staticmethod
    def is_sat(self, res) -> bool:
        raise NotImplementedError("")

    @staticmethod
    def is_unsat(self, res) -> bool:
        raise NotImplementedError("")

    @staticmethod
    def _solver_solve(self, solver, fml):
        raise NotImplementedError("")

    @staticmethod
    def _solver_model(self, solver, res):
        raise NotImplementedError("")

    @staticmethod
    def _model_result(self, solver, model, var, idx):
        raise NotImplementedError("")

    @staticmethod
    def replace_point(expr, ver_vars, point):
        raise NotImplementedError("")

    def verify(self, V_symbolic: SYMBOL, Vdot_symbolic: SYMBOL, **kwargs):
        """
        :param V_symbolic: z3 expr
        :param Vdot_symbolic: z3 expr
        :return:
                found_lyap: True if C is valid
                C: a list of ctx
        """
        found, timed_out = False, False
        fmls = self.constraints_method(self, V_symbolic, Vdot_symbolic)
        results = {}
        solvers = {}
        solver_vars = {}

        for group in fmls:
            for label, condition_vars in group.items():
                if isinstance(condition_vars, tuple):
                    # CBF returns different variables depending on constraint
                    condition, vars = condition_vars
                else:
                    # Other barriers always use only state variables
                    condition = condition_vars
                    vars = self.xs

                s = self.new_solver()
                res, timedout = self.solve_with_timeout(s, condition)
                results[label] = res
                solvers[label] = s
                solver_vars[label] = vars  # todo: select diff vars for input and state
                # if sat, found counterexample; if unsat, C is lyap
                if timedout:
                    logging.info(label + "timed out")
            if any(self.is_sat(res) for res in results.values()):
                break

        ces = {label: [] for label in results.keys()}  # [[] for res in results.keys()]

        if all(self.is_unsat(res) for res in results.values()):
            logging.info("No counterexamples found!")
            found = True
        else:
            for index, o in enumerate(results.items()):
                label, res = o
                if self.is_sat(res):
                    original_point = self.compute_model(
                        vars=solver_vars[label], solver=solvers[label], res=res
                    )
                    logging.info(f"{label}: Counterexample Found: {original_point}")

                    V_ctx = self.replace_point(
                        V_symbolic, solver_vars[label], original_point.numpy().T
                    )
                    Vdot_ctx = self.replace_point(
                        Vdot_symbolic, solver_vars[label], original_point.numpy().T
                    )
                    logging.info("\nV_ctx: {} ".format(V_ctx))
                    logging.info("\nVdot_ctx: {} ".format(Vdot_ctx))

                    ces[label] = self.randomise_counterex(original_point)
                else:
                    logging.info(res)

        return {"found": found, "cex": ces}

    def solve_with_timeout(self, solver, fml):
        """
        :param fml:
        :param solver: z3 solver
        :return:
                res: sat if found ctx
                timedout: true if verification timed out
        """
        try:
            # todo: does it work for solver to set timeout this way?
            solver.set("timeout", max(1, self._solver_timeout * 1000))
        except:
            pass
        logging.debug("Fml: {}".format(fml))
        timer = timeit.default_timer()
        res = self._solver_solve(solver, fml)
        timer = timeit.default_timer() - timer
        timedout = timer >= self._solver_timeout
        return res, timedout

    def compute_model(self, vars, solver, res):
        """
        :param vars: list of solver vars appearing in res
        :param solver: solver
        :return: tensor containing single ctx
        """
        model = self._solver_model(solver, res)
        temp = []
        for i, x in enumerate(vars):
            n = self._model_result(solver, model, x, i)
            temp += [n]

        original_point = torch.tensor(temp)
        return original_point[None, :]

    # given one ctx, useful to sample around it to increase data set
    # these points might *not* be real ctx, but probably close to invalidity condition
    # todo: this is not work of the consolidator?
    def randomise_counterex(self, point):
        """
        :param point: tensor
        :return: list of ctx
        """
        C = []
        # dimensionality issue
        shape = (1, max(point.shape[0], point.shape[1]))
        point = point.reshape(shape)
        for i in range(self.counterexample_n):
            random_point = point + 5 * 1e-4 * torch.randn(shape)
            # if self.inner < torch.norm(random_point) < self.outer:
            C.append(random_point)
        C.append(point)
        return torch.stack(C, dim=1)[0, :, :]


class VerifierZ3(Verifier):
    @staticmethod
    def new_vars(n, base="x"):
        return [z3.Real(base + str(i)) for i in range(n)]

    def new_solver(self):
        return z3.Solver()

    @staticmethod
    def check_type(x) -> bool:
        """
        :param x: any
        :returns: True if z3 compatible, else false
        """
        return contains_object(x, z3.ArithRef)

    @staticmethod
    def replace_point(expr, ver_vars, point):
        """
        :param expr: z3 expr
        :param z3_vars: z3 vars, matrix
        :param ctx: matrix of numerical values
        :return: value of V, Vdot in ctx
        """
        replacements = []
        for i in range(len(ver_vars)):
            try:
                replacements += [(ver_vars[i, 0], z3.RealVal(point[i, 0]))]
            except TypeError:
                replacements += [(ver_vars[i], z3.RealVal(point[i, 0]))]

        replaced = z3.substitute(expr, replacements)

        return z3.simplify(replaced)

    def is_sat(self, res) -> bool:
        return res == z3.sat

    def is_unsat(self, res) -> bool:
        return res == z3.unsat

    def _solver_solve(self, solver, fml):
        solver.add(fml)
        return solver.check()

    def _solver_model(self, solver, res):
        return solver.model()

    def _model_result(self, solver, model, x, i):
        try:
            return float(model[x].as_fraction())
        except AttributeError:
            try:
                return float(model[x].approx(10).as_fraction())
            except AttributeError:
                # no variable in model, eg. input in CBF unfeasible condition. return dummy 0.0
                return 0.0
        except TypeError:
            try:
                return float(model[x[0, 0]].as_fraction())
            except:  # when z3 finds non-rational numbers, prints them w/ '?' at the end --> approx 10 decimals
                return float(model[x[0, 0]].approx(10).as_fraction())

    def solver_fncts(self):
        return FUNCTIONS


def make_verifier(type: VerifierType) -> Type[VerifierZ3]:
    if type == VerifierType.Z3:
        return VerifierZ3
    else:
        raise ValueError(f"Unknown verifier type {type}")
