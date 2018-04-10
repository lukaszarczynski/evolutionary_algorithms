import numpy as np

from random import randrange


def _cyclic_get(child, dictionary, element):
    while element in child:
        if element == dictionary[element]:
            return element
        element = dictionary[element]
    return element


def _copy_remaining(child, p1, p2, chunk_start_idx, chunk_end_idx):
    initial_child_set = set(child)
#     for i, parent_elem in enumerate(p1):
#         if parent_elem not in initial_child_set:
#             child[i] = parent_elem
    child[:chunk_start_idx] = p1[:chunk_start_idx]
    child[chunk_end_idx:] = p1[chunk_end_idx:]

    parent_dict = dict(zip(p1, p2))
    for i, child_elem in enumerate(child):
        if p1[i] in initial_child_set and (i < chunk_start_idx or i >= chunk_end_idx):
            child[i] = _cyclic_get(set(child), parent_dict, p1[i])
    return child


def pmx(parent1, parent2):
    n = len(parent1)
    chunk_start_idx, chunk_end_idx = sorted([randrange(0, n), randrange(0, n)])
    child1 = -np.ones(parent1.shape).astype(np.int64)
    child2 = child1.copy()
    child1[chunk_start_idx:chunk_end_idx] = parent2[chunk_start_idx:chunk_end_idx]
    child2[chunk_start_idx:chunk_end_idx] = parent1[chunk_start_idx:chunk_end_idx]
    child1 = _copy_remaining(child1, parent1, parent2, chunk_start_idx, chunk_end_idx)
    child2 = _copy_remaining(child2, parent2, parent1, chunk_start_idx, chunk_end_idx)
    return child1, child2
