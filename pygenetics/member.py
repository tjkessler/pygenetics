from typing import Union


class Member:

    def __init__(self, params: list, obj_fn_val: Union[int, float]):
        ''' Member object: houses information about a population member's
        currently-used parameters, the return value of the objective function
        using these parameters, and the fitness score resulting from this
        return value

        Args:
            params (list): currently-used parameter values
            obj_fn_val (int, float): value returned by the user-supplied
                objective function
        '''

        self._params = params
        self._obj_fn_val = obj_fn_val
        self._fitness_score = self.calc_fitness(obj_fn_val)

    @staticmethod
    def calc_fitness(obj_fn_val: Union[int, float]) -> float:
        ''' Static method: Member.calc_fitness: Calculates fitness score based
        on objective function value, using the equation:

        fitness = 1 / (1 + ofv)     if ofv >= 0
        fitness = 1 + abs(ofv)      if ofv < 0

        Where `ofv` is the objective function value and `fitness` is the
        resulting fitness score

        Args:
            obj_fn_val (int, float): value obtained from objective function

        Returns:
            float: resulting fitness score
        '''

        if obj_fn_val >= 0:
            return 1 / (obj_fn_val + 1)
        else:
            return 1 + abs(obj_fn_val)
