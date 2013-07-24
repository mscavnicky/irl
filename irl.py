from random import random, randint, sample, choice, uniform

from common import take, accumulate, pairwise
from common import coinflip
from common import std

"""
irl - Iterative Rule Learning

Implementation of algoritm by Aguillar (1993).
"""

class DictionaryData(object):
    "Data provided as a list of dictionaries."
    def __init__(self, data):
        self.data = data

    def count(self):
        return len(self.data)

    def get(self, index):
        return self.data[index]

    def find(self, key):
        return (item.get(key) for item in self.data)

    def match(self, predicates):
        "Find records matching to conjuction of predicates."
        for item in self.data:
            matches = all(predicate.matches(item) for predicate in predicates)
            if matches:
                yield item


class NumericAttribute(object):
    def __init__(self, data, name):
        self.name = name

        values = list(data.find(name))
        self.lower_bound = min(values)
        self.upper_bound = max(values)
        self.range = self.upper_bound - self.lower_bound
        self.std = std(values)

    def __str__(self):
        return "%s: (%f, %f)" % (self.name, self.lower_bound, self.upper_bound)
    def __repr__(self):
        return self.__str__()

class CategoricAttribute(object):
    def __init__(self, data, name):
        self.name = name
        self.values = set(data.find(name))
        self.range = len(self.values)

    def __str__(self):
        return "%s: %s" % (self.name, self.values)
    def __repr__(self):
        return self.__str__()


class NumericPredicate(object):
    "Interval predicate is true if the value of an attribute is in the interval."
    def __init__(self, attribute_name, lower_bound, upper_bound):
        self.attribute_name = attribute_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def matches(self, item):
        value = item[self.attribute_name]
        return self.lower_bound <= value <= self.upper_bound

    def is_empty(self):
        return self.lower_bound > self.upper_bound

    def __str__(self):
        return "%s in (%.3f, %.3f)" % (self.attribute_name, self.lower_bound, self.upper_bound)

class CategoricPredicate(object):
    "Set predicate is true if the values of the attributes belong to the set of values."
    def __init__(self, attribute_name, values):
        self.attribute_name = attribute_name
        self.values = values

    def matches(self, item):
        value = item[self.attribute_name]
        return value in self.values

    def is_empty(self):
        return not self.values

    def __str__(self):
        return "%s in %s" % (self.attribute_name, self.values)

class Rule(object):
    """
    Rule is a conjunction of predicates.
    Predicates are given as a list for each attribute.
    A rule therefore specifies a hypercube in the search space.
    """
    def __init__(self, predicates):
        self.predicates = predicates
        self.fitness = None
        self.evaluation = None

    def is_empty(self):
        "Checks if any of the predicates is empty."
        return any(predicate.is_empty() for predicate in self.predicates)

    def __str__(self):
        rule_str = (predicate.__str__() for predicate in self.predicates)
        return "%.3f - %s - %s" % (self.fitness, self.evaluation, ' ^ '.join(rule_str))
        

class PredicatePopulationGenerator(object):
    def __init__(self, predicate):
        self.predicate = predicate

    "Generates the initial population of rules."
    def generate(self, data, attributes, population_size):
        sample_items = take(population_size, self._sample_items(data))
        return (Rule(list(self._generate_predicates(attributes, item))) for item in sample_items)

    def _sample_items(self, data):
        n = data.count()
        while True:
            item = data.get(randint(0, n-1))
            if self.predicate.matches(item):
                yield item

    def _generate_predicates(self, attributes, item):
        for attribute in attributes:
            yield self._generate_predicate(attribute, item)

    def _generate_predicate(self, attribute, item):
        "Generate predicates for each attribute based on a existing data item."
        value = item.get(attribute.name)
        if isinstance(attribute, NumericAttribute):
            if value:
                lower_bound = value - attribute.range * random() / 2
                upper_bound = value + attribute.range * random() / 2
                return NumericPredicate(attribute.name, lower_bound, upper_bound)
            else:
                return NumericPredicate(attribute.name, attribute.lower_bound, attribute.upper_bound)               
        elif isinstance(attribute, CategoricAttribute):
            if value:
                k = attribute.range * random()
                values = sample(attribute.values, int(k))
                values.append(value)
                return CategoricPredicate(attribute.name, values)
            else:
                return CategoricPredicate(attribute.name, attribute.values)
        else:
            raise TypeError


class RouletteSelection(object):
    """
    Roulette wheel selection operator (generator).
    Does not work with negative fitness values.
    """
    def select(self, population):
        population_fitness = [rule.fitness for rule in population]
        total_fitness = float(sum(population_fitness))
        wheel = list(map(lambda x: x / total_fitness, accumulate(population_fitness)))
        while True:
            u = random()
            i = next(i for i, p in enumerate(wheel) if u < p)
            yield population[i]
    

class TournamentSelection(object):
    "Tournament selection operator (deterministic)."
    def __init__(self, probability):
        self.probability = probability

    def select(self, population):
        while True:
            tournament = sample(population, 2)
            tournament.sort(key=lambda rule: rule.fitness, reverse=True)
            yield tournament[0] if coinflip(self.probability) else tournament[1]


class RuleCrossover(object):
    "Rule crossover as described by Aguilar (2003)."
    def __init__(self, crossover_rate):
        self.crossover_rate = crossover_rate
        self.uniform_crossover_rate = 0.5

    def recombinate(self, population):
        while True:
            rule1, rule2 = take(2, population)
            if coinflip(self.crossover_rate):
                yield self._crossover_rules(rule1, rule2)
                yield self._crossover_rules(rule1, rule2)
            else:
                yield rule1
                yield rule2

    def _crossover_rules(self, rule1, rule2):
        "Recombines two parent rules into a single offspring."
        new_predicates = []
        for predicate1, predicate2 in zip(rule1.predicates, rule2.predicates):
            new_predicates.append(self._crossover_predicates(predicate1, predicate2))
        return Rule(new_predicates)
            
    def _crossover_predicates(self, predicate1, predicate2):
        if isinstance(predicate1, NumericPredicate):
            lower_bound = uniform(predicate1.lower_bound, predicate2.lower_bound)
            upper_bound = uniform(predicate1.upper_bound, predicate2.upper_bound)
            return NumericPredicate(predicate1.attribute_name, lower_bound, upper_bound)
        elif isinstance(predicate1, CategoricPredicate):
            set_difference1 = predicate1.values - predicate2.values
            set_difference2 = predicate2.values - predicate1.values
            for value in set_difference1:
                if coinflip(self.uniform_crossover_rate):
                    predicate2.values.add(value)
                    predicate1.values.remove(value)
            for value in set_difference2:
                if coinflip(self.uniform_crossover_rate):
                    predicate1.values.add(value)
                    predicate2.values.remove(value)
        else:
            raise TypeError

class RuleMutation(object):
    """
    Mutation of a numerical attribute is implemented according to Aguilar (2003).
    This results into a generalization of a rule.
    Mutation of a categorical attribute is slightly changed.
    """
    def __init__(self, mutation_rate):
        self.mutation_rate = mutation_rate

    def mutate(self, population, attributes):
        "Performs in-place mutation of the items in population."
        for rule in population:
            for attribute, predicate in zip(attributes, rule.predicates):
                if coinflip(self.mutation_rate):
                    self._mutate_predicate(predicate, attribute)
            yield rule

    def _mutate_predicate(self, predicate, attribute):
        if isinstance(predicate, NumericPredicate):
            if coinflip(0.5):
                predicate.lower_bound -= 0.05 * attribute.std * random()
            else:
                predicate.upper_bound += 0.05 * attribute.std * random()
        elif isinstance(predicate, CategoricPredicate):
            value = choice(attribute.values)
            if value in predicate.values:
                predicate.values.remove(value)
            else:
                predicate.add(value)
        else:
            raise TypeError


class PredicateFitnessFunction(object):
    def __init__(self, predicate):
        self.predicate = predicate

    def evaluate(self, data, rule):
        correct = incorrect = 0
        matching_items = data.match(rule.predicates)
        for item in matching_items:
            if self.predicate.matches(item):
                correct += 1
            else:
                incorrect += 1

        support = correct + incorrect
        fitness = correct - incorrect
        return fitness, (support, correct, incorrect)


class IterativeRuleLearning(object):
    """
    Initializes IterativeRuleLearning using provided keyword arguments.

    Required arguments:
        'data' - a list of dictionaries containing training data
        'attributes' - a list of Attribute object specifying attributes for rules
        'fitness_function' - a fitness function to evaluate goodness of rules
        'population_generator' - generate the initial population
        'selection_operator' - selection operator for the rules
        'crossover_operators' - a list of crossover operators
        'mutation_operators' - a list of mutation operators        

    Optional arguments:
        'min_support' - minimum percent of data for a rule needs to cover (default 0.01)
        'max_generations' - maximum amount of generations (default 100)
        'population_size' - the size of evolved population (default 10)
        'elitism_level' - number of best individuals to be copied to next generation (default 1)

    Rules competing in the single run of the algorithm all need to use the same attributes.
    However some rules can be given a don't care attribute.
    """
    def __init__(self, data, attributes, **kwargs):
        self.data = data
        self.attributes = attributes
        self.fitness_function = kwargs['fitness_function']
        self.population_generator = kwargs['population_generator']
        self.selection_operator = kwargs['selection_operator']
        self.crossover_operator = kwargs['crossover_operator']
        self.mutation_operator = kwargs['mutation_operator']     

        self.max_generations = kwargs.get('max_generations', 40)
        self.population_size = kwargs.get('population_size', 40)
        self.elitism_level = kwargs.get('elitism_level', 1)

    def update_fitness(self, population):
        for rule in population:
            if not rule.fitness:
                rule.fitness, rule.evaluation = self.fitness_function.evaluate(self.data, rule)

    def best_fitness(self, population):
        best_fitness, best_evaluation = float("-inf"), None
        for rule in population:
            if best_fitness < rule.fitness:
                best_fitness, best_evaluation = rule.fitness, rule.evaluation
        return best_fitness, best_evaluation

    def mine_rule(self):
        population = list(self.population_generator.generate(self.data, self.attributes, self.population_size))

        generation = 0
        while generation < self.max_generations:
            generation += 1
            self.update_fitness(population)
            
            best_fitness, best_evaluation = self.best_fitness(population)
            print 'Generation %d - Best %d, %s.' % (generation,  best_fitness, best_evaluation)

            sorted_population = sorted(population, key=lambda rule: rule.fitness, reverse=True)
            elite_population = take(self.elitism_level, sorted_population)

            new_population = self.selection_operator.select(population)
            new_population = self.crossover_operator.recombinate(new_population)
            new_population = self.mutation_operator.mutate(new_population, self.attributes)

            new_population = (rule for rule in new_population if not rule.is_empty())
            new_population = take(self.population_size - self.elitism_level, new_population)           
            
            new_population.extend(elite_population)
            
            population = new_population

        return population