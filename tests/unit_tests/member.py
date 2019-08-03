import unittest

from pygenetics import Member


class TestMember(unittest.TestCase):

    def test_member(self):

        m = Member([0, 0, 0], 0.0)
        self.assertEqual(m._params, [0, 0, 0])
        self.assertEqual(m._obj_fn_val, 0.0)
        self.assertEqual(m._fitness_score, 1.0)
        self.assertEqual(m.calc_fitness(1.0), 0.5)
        self.assertEqual(m.calc_fitness(-1.0), 2.0)


if __name__ == '__main__':

    unittest.main()
