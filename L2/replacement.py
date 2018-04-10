import numpy as np


def mu_plus_lambda_replacement(objective_values, children_objective_values,
                               current_population, children_population,
                               population_size):
    """replacing the current population by (Mu + Lambda) Replacement"""
    objective_values = np.hstack([objective_values, children_objective_values])
    current_population = np.vstack([current_population, children_population])

    I = np.argsort(objective_values)
    current_population = current_population[I[:population_size], :]
    objective_values = objective_values[I[:population_size]]
    return objective_values, current_population