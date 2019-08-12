# Stdlib. imports
from copy import deepcopy
import logging

# 3rd party imports
from ecnet.utils.data_utils import DataFrame
from ecnet.utils.server_utils import default_config, train_model

# PyGenetics imports
from pygenetics import Population

# Set up logging
logger = logging.Logger(__name__)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    '[%(asctime)s] [%(levelname)s] %(message)s', '%H:%M:%S'
))
logger.addHandler(stream_handler)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


def evaluate_input_vars(var_indices, df):

    # Convert descriptor indices to names
    input_names = [df._input_names[i] for i in var_indices]
    logger.debug('Input descriptor names: {}'.format(input_names))

    # Set the dataset to selected names
    df = deepcopy(df)
    df.set_inputs(input_names)

    # Use default neural network hyper-parameters, 500 learning epochs
    hyperparams = default_config()
    hyperparams['epochs'] = 500

    # Train a neural network, return RMSE of predictions
    rmse = train_model(df.package_sets(), hyperparams, None, 'rmse',
                       validate=False, save=False)
    logger.debug('RMSE: {}'.format(rmse))
    return rmse


def log_best(pop):

    logger.info('Best RMSE: {}'.format(pop.best_ret_val))
    logger.info('Average RMSE: {}'.format(pop.average_ret_val))


def main(database, pop_size, num_desc, num_generations):

    # Import cetane number database, 5305 descriptors from alvaDesc
    df = DataFrame(database)
    logger.info('Loaded data from {}'.format(database))

    # Search space equal to the number of descriptors
    num_input_vars = len(df._input_names)

    # Initialize the population with specified population members
    population = Population(pop_size, evaluate_input_vars, {'df': df}, 8)

    # Add integer values (indices of search space) to optimize
    logger.info('Optimizing training using {} descriptors'.format(num_desc))
    for _ in range(num_desc):
        population.add_param(0, num_input_vars - 1)

    # Initialize the population
    logger.info('Initializing population...')
    population.initialize()
    log_best(population)

    # Run the population for specified number of generations
    for i in range(num_generations):
        logger.info('Generation {}...'.format(i + 1))
        population.next_generation(0.5, 0.1)
        log_best(population)


if __name__ == '__main__':

    # Optimize input variables for cetane number, using 100 population members,
    #   15 descriptors for 5 generations
    main('cn_database_v1.0.csv', 200, 15, 20)
