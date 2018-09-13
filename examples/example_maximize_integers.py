# EXAMPLE SCRIPT: find the combination of 3 integers
#   between 0 and 10 that result in the largest sum

# Import Population object
from pygenetics import Population
# Import maximize selection function
from pygenetics.selection_functions import maximize_best_n


# Cost function; finds the sum of integers in feed_dict
#   (fitness score = sum)
def sum_of_integers(feed_dict, cost_fn_args=None):
    sum = 0
    for integer in feed_dict:
        sum += feed_dict[integer]
    return sum

if __name__ == '__main__':

    # Initialize Population object with population size 10,
    #   cost function of 'sum_of_integers'
    pop = Population(10, sum_of_integers, select_fn=maximize_best_n)

    # Add three integer parameters for each population member
    #   for the genetic algorithm to optimize (randomly
    #   initialized between 0 and 10)
    pop.add_parameter('first_integer', 0, 10)
    pop.add_parameter('second_integer', 0, 10)
    pop.add_parameter('third_integer', 0, 10)

    # Generate initial population with random parameter values
    pop.generate_population()

    # Run the genetic algorithm for 15 generations
    num_generations = 15
    for generation in range(num_generations):
        # Generate the next generation, using top 5 performers,
        #   a mutation rate of 20%, and a maximum mutation amount
        #   of 20% (0.2 * 10, 10 = param max - param min)
        pop.next_generation(5, mut_rate=0.2, max_mut_amt=0.2)

    # Print the fitness score and parameter values for each member
    for idx, member in enumerate(pop.members):
        print('\nMember: ' + str(idx + 1))
        print('Fitness Score: ' + str(member.fitness_score))
        print('Parameters: ' + str(member.feed_dict))
