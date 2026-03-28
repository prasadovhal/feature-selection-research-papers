
import numpy as np

def ensure_non_empty_population(population):
    """Ensure no individual has zero selected features."""
    for i in range(len(population)):
        if np.sum(population[i]) == 0:
            population[i] = np.random.randint(0, 2, size=len(population[i]))
    return population
