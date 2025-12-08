import scenarios.test_scenario as test_scenario
from controllers.proj_controller import findBestChromosome

"""
Runs genetic algorithm and prints results
"""
def run_training(population_size, generation_goal):
    best_chromo = findBestChromosome(population_size, generation_goal)
    print(best_chromo)

"""
Program entry point

Scenarios are essentially drivers that run the game.
Import and run desired scenario below.
"""
if __name__ == "__main__":
    # test_scenario.run()
    run_training(population_size=10, generation_goal=3)
