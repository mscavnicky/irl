"""
irl - Iterative Rule Learning

Slightly modified implementation of an algorithm by Aguillar-Ruiz, Riquelme and 
Toro from Evolutionary Learning of Hiearchical Decision Rules (HIDER).

Iterative Rule Learning is useful for producing classification rules. The 
advantage over decision trees is it does not cover the whole search space, and 
it can capture classes embedded in other classes.
"""

import random
from collections import OrderedDict

from common import take, accumulate, pairwise, coinflip, std


class Dataset(object):
    """Represents a classification dataset including metadata. Stored in a row
    based format, where each row is a dictionary."""

    @staticmethod
    def from_csv(csv_dict_reader, attr_defs):
        """Create dataset from DictReader given the attribute definitions."""
        # Read the file in to a list of dictionaries
        items, classes = [], []
        for row in csv_dict_reader:
            item = OrderedDict()
            class_ = None        
            for attr_name, attr_type in attr_defs:
                if attr_type is 'numeric':
                    item[attr_name] = float(row[attr_name])
                elif attr_type is 'categoric':
                    item[attr_name] = row[attr_name]
                elif attr_type is 'class':
                    class_ = row[attr_name]
                else:
                    raise TypeError
            items.append(item)
            classes.append(class_)

        # Prepare attribute metadata
        attrs = []
        class_attr = None
        for attr_name, attr_type in attr_defs:
            if attr_type is 'numeric':
                values = list(item[attr_name] for item in items)
                lower_bound = min(values)
                upper_bound = max(values)        
                attrs.append(NumericAttribute(attr_name, lower_bound, upper_bound, std(values)))
            elif attr_type is 'categoric':
                categories = set(item[attr_name] for item in items)
                attrs.append(CategorigAttribute(attr_name, categories))
            elif attr_type is 'class':
                class_attr = CategoricAttribute(attr_name, set(classes))
            else:
                raise TypeError

        return Dataset(items, classes, attrs, class_attr)


    def __init__(self, items, classes, attrs, class_attr):
        self.items = items
        self.classes = classes
        self.attrs = attrs
        self.class_attr = class_attr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        """Return a tuple (item, class) with a specified index."""
        return (self.items[index], self.classes[index])

    def sample(self, n):
        """Generate n random items from the dataset"""
        return random.sample(self, n)

    def match(self, rule):
        """Find all records matching given rule."""
        return ((item, class_) for item, class_ in self if rule.matches(item))


class NumericAttribute(object):
    """Represents a real attribute in the dataset. Should be used for the
    integral attributes as well."""
    def __init__(self, name, lower_bound, upper_bound, std):
        self.name = name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.std = std

    @property
    def range(self):
        return self.upper_bound - self.lower_bound

    def __str__(self):
        return "%s: [%f, %f]" % (self.name, self.lower_bound, self.upper_bound)
    def __repr__(self):
        return self.__str__()


class CategoricAttribute(object):
    """Represent a categoric attributes in the dataset."""
    def __init__(self, name, categories):
        self.name = name
        self.categories = categories
        
    @property
    def range(self):
        return len(self.categories)

    def __str__(self):
        return "%s: %s" % (self.name, self.categories)
    def __repr__(self):
        return self.__str__()


class NumericPredicate(object):
    """Interval predicate is true if the value of an attribute is within the
    interval."""
    def __init__(self, attr_name, lower_bound, upper_bound):
        self.attr_name = attr_name
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def matches(self, value):
        return self.lower_bound <= value <= self.upper_bound

    def is_empty(self):
        return self.lower_bound > self.upper_bound

    def __str__(self):
        return "%s in [%.3f, %.3f]" % (self.attr_name, self.lower_bound, self.upper_bound)
    def __repr__(self):
        return self.__str__()


class CategoricPredicate(object):
    """Set predicate is true if the values of the attributes belong to the set
    of values. This predicate could be represented by a bitmap."""
    def __init__(self, attr_name, values):
        self.attr_name = attr_name
        self.values = values

    def matches(self, value):
        return value in self.values

    def is_empty(self):
        return not self.values

    def __str__(self):
        return "%s in {%s}" % (self.attr_name, ', '.join(self.values))
    def __repr__(self):
        return self.__str__()


class Rule(object):
    """Rule is a conjunction of predicates and the class it describes.
    Predicates are given as a list for each attribute. Single rule therefore
    specifies a hypercube in the search space."""
    def __init__(self, predicates, class_):
        self.predicates = predicates
        self.class_ = class_
        self.fitness = None
        self.confus = None

    def matches(self, item):
        """Determines whether rule matches a given instance."""
        return all(pred.matches(val) for pred, val in zip(self.predicates, item.values()))

    def is_empty(self):
        """Checks if any of the predicates is empty."""
        return any(pred.is_empty() for pred in self.predicates)

    def __str__(self):
        antecedent = ' & '.join(str(pred) for pred in self.predicates)
        return "[%f] %s => %s" % (self.fitness, antecedent, self.class_)
    def __repr__(self):
        return self.__str__()
        

class RuleGenerator(object):
    """Generates the initial population of rules by sampling the dataset."""
    def generate(self, dataset, n):
        """Returns a list of n rules randomly generated from the dataset."""
        for item, class_ in dataset.sample(n):
            predicates = [self._generate_predicate(attr, item) for attr in dataset.attrs]
            yield Rule(predicates, class_)

    def _generate_predicate(self, attr, item):
        """Generate predicate for an attribute based on a existing data item."""
        value = item.get(attr.name)
        if isinstance(attr, NumericAttribute):
            lower_bound = value - attr.range * random.random() / 2
            upper_bound = value + attr.range * random.random() / 2
            return NumericPredicate(attr.name, lower_bound, upper_bound)
        elif isinstance(attr, CategoricAttribute):
            k = attr.range * random.random()
            values = random.sample(attr.values, int(k))
            values.append(value)
            return CategoricPredicate(attr.name, values)
        else:
            raise TypeError


class RouletteSelection(object):
    """Roulette wheel selection operator (generator). Does not work with
    negative fitness values."""
    def select(self, population):
        population_fitness = [rule.fitness for rule in population]
        total_fitness = float(sum(population_fitness))
        wheel = list(map(lambda x: x / total_fitness, accumulate(population_fitness)))
        while True:
            u = random()
            i = next(i for i, p in enumerate(wheel) if u < p)
            yield population[i]
    

class TournamentSelection(object):
    """Tournament selection operator (deterministic)."""
    def __init__(self, probability):
        self.probability = probability

    def select(self, population):
        while True:
            tournament = random.sample(population, 2)
            tournament.sort(key=lambda rule: rule.fitness, reverse=True)
            yield tournament[0] if coinflip(self.probability) else tournament[1]


class RuleCrossover(object):
    """Rule crossover as described by Aguilar (2003). We have added the
    condition that the rules to be recombined need to be of the same class.

    For real attribute new min and max are generated from the interval given by
    mins or maxes of their parents. For categoric attributes, a uniform bitwise
    crossover is performed."""
    def __init__(self, crossover_rate):
        self.crossover_rate = crossover_rate

    def recombinate(self, population):
        rule_classes = {}

        while True:
            rule = next(population)
            if rule.class_ in rule_classes:
                rule1, rule2 = rule, rule_classes[rule.class_]
                del rule_classes[rule.class_]
            else:
                rule_classes[rule.class_] = rule
                continue

            if coinflip(self.crossover_rate):
                yield self._crossover_rules(rule1, rule2)
                yield self._crossover_rules(rule1, rule2)
            else:
                yield rule1
                yield rule2

    def _crossover_rules(self, rule1, rule2):
        "Recombines two parent rules into a single offspring."
        new_predicates = []
        for pred1, pred2 in zip(rule1.predicates, rule2.predicates):
            new_predicates.append(self._crossover_predicates(pred1, pred2))
        return Rule(new_predicates, rule1.class_)
            
    def _crossover_predicates(self, pred1, pred2):
        if isinstance(pred1, NumericPredicate):
            lower_bound = random.uniform(pred1.lower_bound, pred2.lower_bound)
            upper_bound = random.uniform(pred1.upper_bound, pred2.upper_bound)
            return NumericPredicate(pred1.attr_name, lower_bound, upper_bound)
        elif isinstance(pred1, CategoricPredicate):
            set_difference1 = pred1.values - pred2.values
            set_difference2 = pred2.values - pred1.values
            for value in set_difference1:
                if coinflip(0.5):
                    pred2.values.add(value)
                    pred1.values.remove(value)
            for value in set_difference2:
                if coinflip(0.5):
                    pred1.values.add(value)
                    pred2.values.remove(value)
        else:
            raise TypeError


class RuleMutation(object):
    """Mutation of a numerical attribute is implemented according to Aguilar
    (2003). This results into a generalization of a rule. Mutation of a
    categorical attribute is slightly changed."""
    def __init__(self, mutation_rate, strength):
        self.mutation_rate = mutation_rate
        self.strength = strength

    def mutate(self, population, attrs):
        """Performs in-place mutation of the items in population."""
        for rule in population:
            for attr, predicate in zip(attrs, rule.predicates):
                if coinflip(self.mutation_rate):
                    self._mutate_predicate(predicate, attr)
            yield rule

    def _mutate_predicate(self, predicate, attr):
        if isinstance(predicate, NumericPredicate):
            if coinflip(0.5):
                predicate.lower_bound -= self.strength * attr.std * random.uniform(-1, 1)
                predicate.lower_bound = max(predicate.lower_bound, attr.lower_bound)
            else:
                predicate.upper_bound += self.strength * attr.std * random.uniform(-1, 1)
                predicate.upper_bound = min(predicate.upper_bound, attr.upper_bound)
        elif isinstance(predicate, CategoricPredicate):
            value = random.choice(attr.categories)
            if value in predicate.values:
                predicate.values.remove(value)
            else:
                predicate.add(value)
        else:
            raise TypeError


class RuleFitnessFunction(object):
    def evaluate(self, dataset, rule):
        correct, incorrect = 0, 0
        matching_items = dataset.match(rule)
        for _, class_ in matching_items:
            if rule.class_ == class_:
                correct += 1
            else:
                incorrect += 1

        support = correct + incorrect
        fitness = correct - 10 * incorrect
        return fitness, (support, correct, incorrect)


class IterativeRuleLearning(object):
    """
    Initializes IterativeRuleLearning using provided keyword arguments.

    Required arguments:
        'dataset' - the classification dataset for the rule mining
        'fitness_func' - a fitness function to evaluate goodness of rules
        'rule_generator' - for generating the initial population
        'selection' - selection operator for the rules
        'crossover' - a crossover operator
        'mutation' - a mutation operator

    Optional arguments:
        'max_generations' - maximum amount of generations (default 100)
        'population_size' - the size of evolved population (default 10)
        'elitism_level' - number of best individuals to be copied to next generation (default 1)

    Rules competing in the single run of the algorithm all need to use the same
    attributes. However some rules can be given a don't care attribute.
    """
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset

        self.fitness_func = kwargs['fitness_func']
        self.rule_generator = kwargs['rule_generator']
        self.selection = kwargs['selection']
        self.crossover = kwargs['crossover']
        self.mutation = kwargs['mutation']     

        self.max_generations = kwargs.get('max_generations', 100)
        self.population_size = kwargs.get('population_size', 50)
        self.elitism_level = kwargs.get('elitism_level', 1)

    def _update_fitness(self, population):
        for rule in population:
            rule.fitness, rule.confus = self.fitness_func.evaluate(self.dataset, rule)

    def mine_rule(self):
        population = list(self.rule_generator.generate(self.dataset, self.population_size))

        generation = 0
        while generation < self.max_generations:
            generation += 1
            self._update_fitness(population)
            
            sorted_population = sorted(population, key=lambda rule: rule.fitness, reverse=True)
            elite_population = take(self.elitism_level, sorted_population)

            new_population = self.selection.select(population)
            new_population = self.crossover.recombinate(new_population)
            new_population = self.mutation.mutate(new_population, self.dataset.attrs)

            new_population = (rule for rule in new_population if not rule.is_empty())
            new_population = take(self.population_size - self.elitism_level, new_population)           
            
            new_population.extend(elite_population)
            
            population = new_population

        return elite_population[0]