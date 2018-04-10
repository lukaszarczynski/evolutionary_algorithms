import numpy as np

def reverse_sequence_mutation(population):
    a = np.random.choice(len(population), 2, False)
    i, j = a.min(), a.max()
    q = population.copy()
    q[i:j + 1] = q[i:j + 1][::-1]
    return q


def transposition_mutation(population):
    i, j = np.random.choice(len(population), 2, False)
    q = population.copy()
    q[i], q[j] = population[j], population[i]
    return q