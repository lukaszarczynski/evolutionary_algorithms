from random import random

import numpy as np


def pbil(evaluation_function, population_size, chromosome_length, termination_condition,
         learning_factor, mutation_probability, mutation_factor):
    best_specimens = []
    model_chromosome = np.array([0.5] * chromosome_length)
    population = random_population(model_chromosome, population_size)
    scores = population_evaluation(population, evaluation_function)
    while not termination_condition(evaluation_function, best_specimens):
        best = best_individual(population, scores)
        best_specimens.append(best)
        model_chromosome = update_model(model_chromosome, best, learning_factor)
        model_chromosome = mutation(model_chromosome, mutation_probability, mutation_factor)
        population = random_population(model_chromosome, population_size)
        scores = population_evaluation(population, evaluation_function)
    return model_chromosome, np.array(best_specimens)


def binary_random(probability):
    return random() < probability


def random_individual(chromosome_model):
    return np.random.random(len(chromosome_model)) < chromosome_model
    # return np.array([binary_random(probability) for probability in chromosome_model])


def random_population(model_chromosome, population_size):
    return np.random.random((population_size, len(model_chromosome))) < model_chromosome[np.newaxis]
    # return np.array([random_individual(model_chromosome) for _ in range(population_size)])


def population_evaluation(population, evaluation_function):
    return evaluation_function(population)
    # return [evaluation_function(specimen) for specimen in population]


def update_model(model_chromosome, best, learning_factor):
    return (model_chromosome * (1 - learning_factor)
            + best * learning_factor)
    # new_model = []
    # for model_probability, best_result in zip(model_chromosome, best):
    #     new_model.append(model_probability * (1 - learning_factor)
    #                      + best_result * learning_factor)
    # return np.array(new_model)


def mutation(model_chromosome, mutation_probability, mutation_factor):
    new_model = []
    for model_probability in model_chromosome:
        if random() < mutation_probability:
            new_model.append(model_probability * (1 - mutation_factor)
                             + binary_random(0.5) * mutation_factor)
        else:
            new_model.append(model_probability)
    return np.array(new_model)


def best_individual(population, scores):
    best_individual_id = np.argmax(scores)
    return population[best_individual_id]


if __name__ == "__main__":
    def iterations_limit(max_iterations):
        def stop_iteration(_, best_specimens):
            current_iteration = len(best_specimens)
            return current_iteration >= max_iterations
        return stop_iteration

    def one_max(population):
        return np.sum(population, axis=1)

    model, best_specimens = pbil(one_max, 125, 100, iterations_limit(1000),
                                 0.01, 0.05, 0.01)

    print(model)
    print(population_evaluation(best_specimens, one_max))
