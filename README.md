# PyGenetics

[![GitHub version](https://badge.fury.io/gh/tjkessler%2FPyGenetics.svg)](https://badge.fury.io/gh/tjkessler%2FPyGenetics)
[![PyPI version](https://badge.fury.io/py/pygenetics.svg)](https://badge.fury.io/py/pygenetics)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://raw.githubusercontent.com/TJKessler/PyGenetics/master/LICENSE.txt)

PyGenetics is an open source genetic algorithm framework designed to optimize sets of parameters for arbitrary cost functions.

## Installation
### Prerequisites:
- Have Python 3.X installed
- Have the ability to install Python packages

### Method 1: pip
If your Python install contains pip:
```
pip install pygenetics
```
Note: if multiple Python releases are installed on your system (e.g. 2.7 and 3.6), you may need to execute the correct version of pip. For Python 3.X, change **pip install pygenetics** to **pip3 install pygenetics**.

### Method 2: From source
- Download the PyGenetics repository, navigate to the download location on the command line/terminal, and execute:
```
python setup.py install
```

To upgrade your version of PyGenetics to the latest release:
```
pip install --upgrade pygenetics
```

## Usage
To get started, import the **Population** object:
```python
from pygenetics import Population
```
Now we need a cost function with parameters to optimize. PyGenetics only requires that the function accepts a **feed_dict** argument, a dictionary of parameter names and corresponding values, additional arbitrary arguments that are used by the cost function **cost_fn_args** (can be in any form, list, dict, etc, depending on the functionality of the supplied cost function), and that it returns a numerical measurement of the function's performance (its **fitness score**). For example, let's create a function to return the sum of all integers found in the **feed_dict**:
```python
def sum_of_integers(feed_dict, cost_fn_args=None):
    sum = 0
    for integer in feed_dict:
        sum += feed_dict[integer]
    return sum
```
Now let's initialize a Population of size 100 to optimize our function:
```python
pop = Population(100, sum_of_integers)
```
By default, the genetic algorithm will **minimize** the supplied cost function; you may supply a custom selection function with:
```python
pop = Population(100, sum_of_integers, select_fn=my_selection_function)
```
Essentially, a supplied selection function should accept a list of unordered population members, and return an ordered list of population members - for some inspiration, view our [pre-built selection functions](https://github.com/tjkessler/PyGenetics/blob/master/pygenetics/selection_functions.py).

So our population is defined, but we still need to add some parameters to optimize. For our integer summation cost function, let's add three integers randomly initialized between 0 and 10:
```python
pop.add_parameter('first_integer', 0, 10)
pop.add_parameter('second_integer', 0, 10)
pop.add_parameter('third_integer', 0, 10)
```
And generate the initial population of 100 members, each with three random integers between 0 and 10:
```python
pop.generate_population()
```
Let's take a look at the population's average fitness:
```python
>>> print(pop.fitness)
0.0832
```
Our goal is to minimize the supplied parameters which increases fitness, so we want this value to be as close to 1 as possible. Let's repopulate our population for 6 generations and print the average population fitness after each generation:
```python
>>> for _ in range(6):
>>>     pop.next_generation()
>>>     print(pop.fitness)
    
0.0832
0.0945
0.1081
0.1100
0.125
0.1289
```

We can also view the average value returned by the supplied cost function:
```python
>>> for _ in range(6):
>>>     pop.next_generation()
>>>     print(pop.ave_cost_fn_val)

12.2
10.0
8.7
8.3
7.1
6.9
```

And the median value returned by the supplied cost function:
```python
>>> for _ in range(6):
>>>     pop.next_generation()
>>>     print(pop.med_cost_fn_val)

11.5
9.5
7.5
8.0
7.0
7.0
```

We can introduce a mutation rate (chance for any of a new population member's parameters to mutate) and a maximum mutation amount (allowed percentage change of the parameter, within the bounds of the parameter's minimum and maximum allowed values):
```python
pop.next_generation(mut_rate=0.05, max_mut_amt=0.2)
```
And change the distribution of chosen parent probabilities (reverse logspace proportions):
```python
pop.next_generation(log_base=20)
```
A higher 'log_base' results in members ordered first by the selection function to have a higher chance of being chosen as parents for the next generation. The default value for 'log_base' is 10.

We can print the population's average parameter values:
```python
>>> print(pop.parameters)
{'first_integer': 0.0, 'second_integer': 0.0, 'third_integer': 0.0}
```

The fitness, cost function value and parameters for the best performing member seen:
```python
>>> print(pop.best_fitness, pop.best_cost_fn_val, pop.best_parameters)
1.0 0 {'first_integer': 0, 'second_integer': 0, 'third_integer': 0}
```

We can also look at the parameter values for an individual population member:
```python
>>> print(pop.members[0].parameters)
{'first_integer': 0, 'second_integer': 0, 'third_integer': 0}
```
And its fitness score and cost function value:
```python
>>> print(pop.members[0].fitness_score, pop.members[0].cost_fn_val)
1.0 0
```
For downloadable example python scripts, visit our [examples directory](https://github.com/tjkessler/PyGenetics/tree/master/examples).

## Contributing, Reporting Issues and Other Support:

To contribute to PyGenetics, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com).
