from random import uniform
from typing import Union


class Parameter:

    def __init__(self, min_val: Union[int, float], max_val: Union[int, float],
                 restrict: bool = True):
        ''' Parameter object: houses information about a user-supplied
        parameter including data type, minimum/maximum initialization values,
        and whether the parameter is limited to [min_val, max_val] when
        mutating

        Args:
            min_val (int, float): minimum value allowed for the parameter's
                initialization
            max_val (int, float): maximum value allowed for the parameter's
                initialization
            restrict (bool): if `True`, parameter mutations must be within
                [min_val, max_val]
        '''

        if type(min_val) != type(max_val):
            raise ValueError('Supplied min_val is not the same type as '
                             'supplied max_val: {}, {}'.format(
                                 type(min_val), type(max_val)
                             ))

        if type(min_val) not in [int, float]:
            raise ValueError('Unsupported data type for Parameter, {}; use {}'
                             .format(type(min_val), [int, float]))

        self._min_val = min_val
        self._max_val = max_val
        self._dtype = type(min_val)
        self._restrict = restrict

    @property
    def rand_val(self) -> Union[int, float]:
        ''' Returns a random value X in range [min_val, max_val] using the
        equation:

        X = min_val + rand(0, 1) * (max_val - min_val)
        '''

        return self._dtype(self._min_val + uniform(0, 1) *
                           (self._max_val - self._min_val))

    def mutate(self, curr_value: Union[int, float]) -> Union[int, float]:
        ''' Parameter.mutate: mutates current parameter value by using the
        equation:

        V = X + rand(-1, 1) * (X - Xrand)

        Where V is the new value, X is the current value and Xrand is a random
        parameter value

        Args:
            curr_value (int, float): current parameter value

        Returns:
            int, float: mutated parameter value
        '''

        new_value = self._dtype(
            curr_value + uniform(-1, 1) * (curr_value - self.rand_val)
        )

        if self._restrict:
            if new_value > self._max_val:
                new_value = self._max_val
            elif new_value < self._min_val:
                new_value = self._min_val

        if new_value == curr_value:
            if self._dtype is int and self._max_val - self._min_val <= 2:
                pass
            else:
                return self.mutate(curr_value)

        return new_value
