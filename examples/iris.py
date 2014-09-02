from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))

import csv
from irl import *


if __name__ == '__main__':
    attrs = [
        ('sepal_length', 'numeric'),
        ('sepal_width', 'numeric'),
        ('petal_length', 'numeric'),
        ('petal_width', 'numeric'),
        ('class', 'class')
    ]

    with open('examples/iris.txt', 'rb') as fread:
        reader = csv.DictReader(fread, delimiter=',', quoting=csv.QUOTE_NONE)
        dataset=Dataset.from_csv(reader, attrs)

    irl = IterativeRuleLearning(
        dataset,
        rule_generator=RuleGenerator(),
        fitness_func=RuleFitnessFunction(),
        selection=TournamentSelection(0.7),
        crossover=RuleCrossover(0.2),
        mutation=RuleMutation(0.1, 0.5)
    )

    print irl.mine_rule()