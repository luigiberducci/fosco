import unittest


from cegis.common.consts import VerifierType
from cegis.verifier import make_verifier, Verifier, SYMBOL


class TestVerifier(unittest.TestCase):

    def test_simple_constraints(self):
        import z3

        verifier_fn = make_verifier(type=VerifierType.Z3)

        def constraint_gen(verif: Verifier, C: SYMBOL, dC: SYMBOL):
            yield {"sat": C >= 0.0}

        def constraint_gen2(verif: Verifier, C: SYMBOL, dC: SYMBOL):
            yield {"unsat": z3.And(C >= 0.0, C < 0)}

        vars = verifier_fn.new_vars(n=1)
        verifier = verifier_fn(solver_vars=vars, constraints_method=constraint_gen)
        verifier2 = verifier_fn(solver_vars=vars, constraints_method=constraint_gen2)

        C = vars[0] + 1.0
        dC = vars[0] + 6.0
        results = verifier.verify(V_symbolic=C, dC=dC)
        results2 = verifier2.verify(V_symbolic=C, dC=dC)

        self.assertTrue(len(results["cex"]["sat"]) > 0, "expected counterexample for any x > -1, got none")
        self.assertTrue(len(results2["cex"]["unsat"]) == 0, f"expected no counterexample, got {results2['cex']['unsat']}")


