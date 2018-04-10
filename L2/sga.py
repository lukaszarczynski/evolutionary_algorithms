import time
import numpy as np

from L2.replacement import mu_plus_lambda_replacement


class SGA:
    def __init__(self, objective_function, chromosome_length,
                 mutation, genetic_operator, replacement=mu_plus_lambda_replacement):
        self.objective_function = objective_function
        self.chromosome_length = chromosome_length
        self.mutation_strategy = mutation
        self.genetic_operator = genetic_operator
        self.replacement_strategy = replacement

        self.population_size = None
        self.number_of_offspring = None
        self.number_of_offspring = None
        self.crossover_probability = None
        self.mutation_probability = None
        self.number_of_iterations = None

    def initial_population(self):
        current_population = np.zeros((self.population_size, self.chromosome_length), dtype=np.int64)
        for i in range(self.population_size):
            current_population[i, :] = np.random.permutation(self.chromosome_length)
        return current_population

    def evaluate_population(self, population):
        objective_values = np.zeros(self.population_size)
        for i in range(self.population_size):
            objective_values[i] = self.objective_function(population[i, :])
        return objective_values

    def mutate_population(self, children_population):
        for i in range(self.number_of_offspring):
            if np.random.random() < self.mutation_probability:
                children_population[i, :] = self.mutation_strategy(children_population[i, :])
        return children_population

    @staticmethod
    def update_best(best, objective_values, best_objective_value, current_population, iteration):
        best[iteration] = objective_values[0]
        if best_objective_value < objective_values[0]:
            best_objective_value = objective_values[0]
        best_chromosome = current_population[0, :]
        return best_objective_value, best_chromosome

    @staticmethod
    def print_stats(objective_values, start_time, iteration):
        print('%3d %14.8f %12.8f %12.8f %12.8f %12.8f' % (
            iteration, time.time() - start_time,
            objective_values.min(), objective_values.mean(),
            objective_values.max(), objective_values.std()))

    def crossover(self, current_population, parent_indices):
        children_population = np.zeros((self.number_of_offspring, self.chromosome_length), dtype=np.int64)
        for i in range(int(self.number_of_offspring / 2)):
            if np.random.random() < self.crossover_probability:
                children_population[2 * i, :], children_population[2 * i + 1, :] = self.genetic_operator(
                    current_population[parent_indices[2 * i], :].copy(),
                    current_population[parent_indices[2 * i + 1], :].copy()
                )
            else:
                children_population[2 * i, :], children_population[2 * i + 1, :] = (
                    current_population[parent_indices[2 * i], :].copy(),
                    current_population[parent_indices[2 * i + 1]].copy())
        if np.mod(self.number_of_offspring, 2) == 1:
            children_population[-1, :] = current_population[parent_indices[-1], :]
        return children_population

    def parent_selection(self, objective_values):
        fitness_values = objective_values.max() - objective_values
        if fitness_values.sum() > 0:
            fitness_values = fitness_values / fitness_values.sum()
        else:
            fitness_values = np.ones(self.population_size) / self.population_size
        parent_indices = np.random.choice(self.population_size, self.number_of_offspring, True, fitness_values
                                          ).astype(np.int64)
        return parent_indices

    def set_parameters(self, population_size, number_of_offspring, number_of_iterations,
                       crossover_probability, mutation_probability):
        self.population_size = population_size
        if number_of_offspring is None:
            self.number_of_offspring = population_size
        else:
            self.number_of_offspring = number_of_offspring
        self.number_of_offspring = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.number_of_iterations = number_of_iterations

    def evolve(self, population_size=500, number_of_offspring=None, number_of_iterations=250,
                crossover_probability=0.95, mutation_probability=0.25):
        self.set_parameters(population_size, number_of_offspring, number_of_iterations,
                            crossover_probability, mutation_probability)

        start_time = time.time()

        best_objective_value = np.Inf
        best_chromosome = np.zeros((1, self.chromosome_length))

        current_population = self.initial_population()

        objective_values = self.evaluate_population(current_population)

        best = np.zeros(self.number_of_iterations)

        for iteration in range(self.number_of_iterations):
            parent_indices = self.parent_selection(objective_values)

            children_population = self.crossover(current_population, parent_indices)

            children_population = self.mutate_population(children_population)

            children_objective_values = self.evaluate_population(children_population)

            objective_values, current_population = self.replacement_strategy(
                objective_values, children_objective_values,
                current_population, children_population,
                self.population_size
            )

            best_objective_value, best_chromosome = SGA.update_best(best, objective_values, best_objective_value,
                                                                    current_population, iteration)

            SGA.print_stats(objective_values, start_time, iteration)
        return best, best_chromosome
