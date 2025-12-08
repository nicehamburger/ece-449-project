import scenarios.test_scenario as test_scenario
from controllers.proj_controller import findBestChromosome
import numpy as np

"""
Runs genetic algorithm and writes the best chromosome to a file
"""
def run_training(population_size, generation_goal):
    best_chromo = findBestChromosome(population_size, generation_goal)
    best_chromo_values = [gene.value for gene in best_chromo.gene_list]
    np.savetxt("best_chromosome.txt", best_chromo_values, delimiter=",", fmt='%.5f')


"""
Program entry point

Scenarios are essentially drivers that run the game.
Import and run desired scenario below.
"""
if __name__ == "__main__":
    # test_scenario.run()
    run_training(population_size=30, generation_goal=10)
