import unittest

from pygenetics import Parameter


class TestParameter(unittest.TestCase):

    def test_init(self):

        p = Parameter(0, 10)
        self.assertEqual(p._min_val, 0)
        self.assertEqual(p._max_val, 10)
        self.assertEqual(p._dtype, int)
        self.assertTrue(p._restrict)

        p = Parameter(0.0, 10.0)
        self.assertEqual(p._dtype, float)

        p = Parameter(0.0, 10.0, False)
        self.assertFalse(p._restrict)

        with self.assertRaises(ValueError):
            p = Parameter(0, 10.0)
        with self.assertRaises(ValueError):
            p = Parameter('a', 'b')

    def test_rand_val(self):

        p = Parameter(0.0, 10.0)
        for _ in range(100):
            rv = p.rand_val
            self.assertGreaterEqual(rv, 0.0)
            self.assertLessEqual(rv, 10.0)

    def test_mutate(self):

        p = Parameter(0.0, 10.0)
        curr_val = 5.0
        mut_val = p.mutate(curr_val)
        self.assertNotEqual(curr_val, mut_val)

        p = Parameter(0.0, 10.0, False)
        curr_val = 5.0
        outside_bounds = False
        for _ in range(1000000):
            curr_val = p.mutate(curr_val)
            if curr_val > 10.0:
                outside_bounds = True
                break
            if curr_val < 0.0:
                outside_bounds = True
                break
        self.assertTrue(outside_bounds)


if __name__ == '__main__':

    unittest.main()
