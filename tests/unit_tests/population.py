import unittest

from pygenetics import Population


def objective_function(params):
    return sum(params)


def objective_function_kwargs(params, my_kwarg):
    return sum(params) + my_kwarg


class TestPopulation(unittest.TestCase):

    def test_init(self):

        p = Population(10, objective_function)
        self.assertEqual(p._pop_size, 10)
        self.assertEqual(p._obj_fn, objective_function)

        kwargs = {'my_kwarg', 2}
        p = Population(10, objective_function_kwargs, kwargs)
        self.assertEqual(p._obj_fn_args, kwargs)

        p = Population(10, objective_function, num_processes=8)
        self.assertEqual(p._num_processes, 8)

    def test_exceptions(self):

        with self.assertRaises(ReferenceError):
            p = Population(10, None)
        with self.assertRaises(ValueError):
            p = Population(1, objective_function)

        p = Population(10, objective_function)
        with self.assertRaises(RuntimeError):
            p.initialize()
        with self.assertRaises(RuntimeError):
            p.next_generation()
        p.add_param(0, 10)
        p.initialize()
        with self.assertRaises(RuntimeError):
            p.add_param(0, 10)
        with self.assertRaises(ValueError):
            p.next_generation(p_crossover=1.1)
        with self.assertRaises(ValueError):
            p.next_generation(p_crossover=-0.1)
        with self.assertRaises(ValueError):
            p.next_generation(p_mutation=1.1)
        with self.assertRaises(ValueError):
            p.next_generation(p_mutation=-0.1)

    def test_none_property(self):

        p = Population(10, objective_function)
        self.assertEqual(p.best_fitness, None)
        self.assertEqual(p.best_ret_val, None)
        self.assertEqual(p.best_params, None)
        self.assertEqual(p.average_fitness, None)
        self.assertEqual(p.average_ret_val, None)

    def test_add_parameter(self):

        p = Population(10, objective_function)
        p.add_param(0, 1)
        p.add_param(2, 3)
        p.add_param(4, 5)
        self.assertEqual(p._params[0]._min_val, 0)
        self.assertEqual(p._params[0]._max_val, 1)
        self.assertEqual(p._params[1]._min_val, 2)
        self.assertEqual(p._params[1]._max_val, 3)
        self.assertEqual(p._params[2]._min_val, 4)
        self.assertEqual(p._params[2]._max_val, 5)
        self.assertEqual(p._params[0]._dtype, int)
        p.add_param(0.0, 10.0)
        self.assertEqual(p._params[3]._dtype, float)
        self.assertTrue(p._params[3]._restrict)
        p.add_param(0.0, 10.0, False)
        self.assertFalse(p._params[4]._restrict)

    def test_initialize(self):

        p = Population(20, objective_function)
        p.add_param(0, 10)
        p.add_param(0, 10)
        p.initialize()
        self.assertEqual(len(p._members), 20)

    def test_get_stats(self):

        p = Population(10, objective_function)
        p.add_param(0, 0)
        p.add_param(0, 0)
        p.initialize()
        self.assertEqual(p.best_fitness, 1)
        self.assertEqual(p.best_ret_val, 0)
        self.assertEqual(p.best_params, [0, 0])
        self.assertEqual(p.average_fitness, 1)
        self.assertEqual(p.average_ret_val, 0)

    def test_kwargs(self):

        p = Population(10, objective_function_kwargs, {'my_kwarg': 2})
        p.add_param(0, 0)
        p.add_param(0, 0)
        p.initialize()
        self.assertEqual(p.best_ret_val, 2)

    def test_next_generation(self):

        p = Population(100, objective_function)
        p.add_param(0, 10)
        p.add_param(0, 10)
        p.add_param(0, 10)
        p.initialize()
        for _ in range(100):
            p.next_generation()
        self.assertEqual(p.best_fitness, 1)
        self.assertEqual(p.best_ret_val, 0)
        self.assertEqual(p.best_params, [0, 0, 0])

    def test_multiprocessing(self):

        p = Population(10, objective_function, num_processes=4)
        self.assertEqual(p._num_processes, 4)
        p.add_param(0, 10)
        p.add_param(0, 10)
        p.initialize()
        p.next_generation()


if __name__ == '__main__':

    unittest.main()
