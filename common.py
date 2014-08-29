"""
Common helper functions.
"""

import operator
import random

from math import sqrt
from itertools import islice, izip, tee

def std(seq):    
    n = len(seq)
    mean = sum(seq) / n
    std =  sqrt(sum((x - mean)**2 for x in seq) / n)
    return std

def take(n, iterable):
    "Return first n items of the iterable as a list."
    return list(islice(iterable, n))

def accumulate(iterable, func=operator.add):
    "Return running totals."
    iterator = iter(iterable)
    total = next(iterator)
    yield total
    for element in iterator:
        total = func(total, element)
        yield total

def pairwise(iterable):
    "Produces pairs of consecutive elements in an iterable."
    a, b = tee(iterable)
    next(b, None)
    return izip(a, b)

def coinflip(probability):
    "Flips a biased coined with a given probablity."
    return random.random() < probability