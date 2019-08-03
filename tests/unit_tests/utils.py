import unittest

import pygenetics.utils as pyg_utils
from pygenetics import Member, Parameter


class TestUtils(unittest.TestCase):

    def test_calc_cdf_vals(self):

        members = [
            Member([0, 0, 0], 1.0),
            Member([0, 0, 0], 2.0),
            Member([0, 0, 0], 3.0)
        ]
        cdf_vals = pyg_utils.calc_cdf_vals(members)
        self.assertEqual(len(cdf_vals), 3)
        self.assertEqual(cdf_vals[2], 1.0)

    def test_call_obj_fn(self):

        def obj_fn(params):
            return sum(params)

        def obj_fn_kw(params, my_kwarg):
            return sum(params) + my_kwarg

        param_vals = [2, 2, 2]

        ret_params, result = pyg_utils.call_obj_fn(param_vals, obj_fn, {})
        self.assertEqual(6, result)
        self.assertEqual(param_vals, ret_params)

        fn_args = {'my_kwarg': 2}
        ret_params, result = pyg_utils.call_obj_fn(
            param_vals, obj_fn_kw, fn_args
        )
        self.assertEqual(8, result)
        self.assertEqual(param_vals, ret_params)

    def test_determine_best_member(self):

        members = [
            Member([0, 0, 0], 0.0),
            Member([1, 1, 1], 1.0),
            Member([2, 2, 2], 2.0)
        ]
        b_fitness, b_ret_val, b_params = pyg_utils.determine_best_member(
            members
        )
        self.assertEqual(b_fitness, 1.0)
        self.assertEqual(b_ret_val, 0.0)
        self.assertEqual(b_params, [0, 0, 0])

    def test_mutate_params(self):

        param_vals = [5, 5, 5]
        params = [
            Parameter(0, 10),
            Parameter(0, 10),
            Parameter(0, 10)
        ]
        new_vals = pyg_utils.mutate_params(param_vals, params, 0.0)
        self.assertEqual(param_vals, new_vals)
        new_vals = pyg_utils.mutate_params(param_vals, params, 1.0)
        self.assertNotEqual(param_vals, new_vals)

    def test_perform_crossover(self):

        p1 = [0, 0]
        p2 = [1, 1]
        p1_new, p2_new = pyg_utils.perform_crossover(p1, p2)
        self.assertEqual(p1_new, [0, 1])
        self.assertEqual(p2_new, [1, 0])

        p1 = [0, 0, 0, 0]
        p2 = [1, 1, 1, 1]
        p1_new, p2_new = pyg_utils.perform_crossover(p1, p2)
        possible_combos = [
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ]
        self.assertIn(p1_new, possible_combos)
        self.assertIn(p2_new, possible_combos)


if __name__ == '__main__':

    unittest.main()
