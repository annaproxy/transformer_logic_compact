"""
Based on the Temporal logic code https://github.com/reactive-systems/deepltl
Will have to install pyaiger "pip install py-aiger-sat"
"""
import aiger_sat
import aiger
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd

spot_to_pyaiger_dict = {
    "1": "True",
    "0": "False",
    "!": "neg",
    "<->": "eq",
    "xor": "xor",
    "&": "and",
    "|": "or",
}


class Generator:
    def __init__(
        self,
        num_variables=10,
        bool_constants=["True", "False"],
        unary_operators=["neg"],
        binary_operators=["or", "and", "xor", "eq"],
    ):
        self.num_variables = num_variables
        self.variables = ["var%02d" % v for v in range(self.num_variables)]
        self.bool_constants = bool_constants.copy()
        self.unary_operators = unary_operators.copy()
        self.binary_operators = binary_operators.copy()

    def to_expression(self, token_sequence):
        token_sequence = iter(token_sequence)
        elem = next(token_sequence, None)
        if elem is None:
            raise ValueError("Sequence ends before expression is complete.")
        if elem in self.variables:
            return aiger.atom(elem)
        if elem in self.bool_constants:
            return aiger.atom(elem == "True")
        if elem in self.unary_operators:
            assert elem == "neg"
            return ~self.to_expression(token_sequence)
        assert elem in self.binary_operators, "Unknown op: %s" % elem
        left = self.to_expression(token_sequence)
        right = self.to_expression(token_sequence)
        if elem == "or":
            return left | right
        if elem == "and":
            return left & right
        if elem == "xor":
            return left ^ right
        if elem == "eq":
            return left == right
        raise ValueError("Should not reach this point")


def get_assignment(lst):
    if len(lst) % 2 != 0:
        # Model messed up the generation
        return None
    ass_it = iter(lst)
    ass_dict = {}
    for var in ass_it:
        val = next(ass_it)
        if val == "True" or val == "1":
            ass_dict[var] = True
        elif val == "False" or val == "0":
            ass_dict[var] = False
        else:
            # Model messed up
            return None
    s = [f"{var}={val}" for (var, val) in ass_dict.items()]
    return ass_dict, " ".join(s)


def spot_to_pyaiger(token_list):
    res = []
    for token in token_list:
        if token in spot_to_pyaiger_dict:
            res.append(spot_to_pyaiger_dict[token])
        else:
            n = ord(token) - 97
            if n >= 26:
                raise ValueError()
            res.append(f"var{n:02}")
    return res


def is_model(input_formula: str, assignment: str) -> bool:
    # Input idx is ignored
    generator = Generator()
    formula_pyaiger = spot_to_pyaiger(input_formula.split())
    a = get_assignment(spot_to_pyaiger(assignment.split()))
    if a is not None:
        assignment_pyaiger, _ = a
    else:
        return None
    formula = generator.to_expression(formula_pyaiger)
    solver = aiger_sat.SolverWrapper()
    solver.add_expr(~formula)
    try:
        return not solver.is_sat(assumptions=assignment_pyaiger)
    except:
        # Model used unknown variables or fucked up in another way
        return None


if __name__ == "__main__":
    print(is_model("xor a b", "a 0 b 0"))
    print(is_model("& & a b c", "a 1 b 1 c 1"))
