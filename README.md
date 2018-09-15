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
- Download the PyGenetics repository, navigate to the download location on the command line/terminal, and execute 
**"python setup.py install"**. 

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
Essentially, a supplied selection function should accept a list of unordered population members, how many population members to select, and return an ordered list of population members - for some inspiration, view our [pre-built selection functions](https://github.com/tjkessler/PyGenetics/blob/master/pygenetics/selection_functions.py).

So our population is defined, but we still need to add some parameters to optimize. For our integer summation cost function, let's add three integers randomly initialized between 0 and 10:
```python
pop.add_parameter('first_integer', 0, 10)
pop.add_parameter('second_integer', 0, 10)
pop.add_parameter('third_integer', 0, 10)
```
And generate the initial population of 100 members, each with three random integers between 1 and 10:
```python
pop.generate_population()
```
Let's take a look at the population's average fitness:
```python
print(pop.fitness)
>>> 15.52
```
Our goal is to minimize the supplied parameters, so we want this value to be as close to 0 as possible. Let's repopulate our population for 6 generations, selecting the best 50 population members from each generation to produce the next generation's 100:
```python
for _ in range(6):
    pop.next_generation(50)
    print(pop.fitness)
    
>>> 9.78
>>> 5.71
>>> 2.43
>>> 0.91
>>> 0.1
>>> 0.0
```
Our parameters look optimized - here are the population's average parameter values:
```python
print(pop.param_vals)
>>> {'first_integer': 0.0, 'second_integer': 0.0, 'third_integer': 0.0}
```
We can also look at the parameter values for an individual population member:
```python
print(pop.members[0].param_vals)
>>> {'first_integer': 0, 'second_integer': 0, 'third_integer': 0}
```
And its fitness score:
```python
print(pop.members[0].fitness_score)
>>> 0
```
For downloadable example python scripts, visit our [examples directory](https://github.com/tjkessler/PyGenetics/tree/master/examples).

## Contributing, Reporting Issues and Other Support:

To contribute to PyGenetics, make a pull request. Contributions should include tests for new features added, as well as extensive documentation.

To report problems with the software or feature requests, file an issue. When reporting problems, include information such as error messages, your OS/environment and Python version.

For additional support/questions, contact Travis Kessler (travis.j.kessler@gmail.com).
