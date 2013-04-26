import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

import csv
from irl import *

def read_iris():
    with open('%s/%s' % (script_dir, 'iris.txt'), 'rb') as iris_file:
        attributes = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
        reader = csv.DictReader(iris_file, attributes, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
            yield {key: float(value) for key, value in row.iteritems()}

if __name__ == '__main__':
    data=DictionaryData(list(read_iris()))

    attributes=[
        NumericAttribute(data, 'sepal_length'), 
        NumericAttribute(data, 'sepal_width'), 
        NumericAttribute(data, 'petal_length'), 
        NumericAttribute(data, 'petal_width')
    ]

    irl = IterativeRuleLearning(
        data, attributes,     
        population_generator=PopulationGenerator(),
        fitness_function=AttributeFitnessFunction('class', 1),
        selection_operator=TournamentSelection(0.7),
        crossover_operator=RuleCrossover(0.1),
        mutation_operator=RuleMutation(0.1)
    )

    irl.mine_rule()