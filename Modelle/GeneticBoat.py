import numpy as np
import logging
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from datetime import datetime, timedelta
from Modelle.GreedyBoat import GreedyBoat
from Modelle.OpenDriftPlastCustom import OpenDriftPlastCustom
import random


class GeneticBoatArray(LagrangianArray):
    variables = LagrangianArray.add_variables([
        ('current_drift_factor',
         {'dtype': np.float32, 'units': '1', 'description': 'For compatibility', 'default': 10.0}),
        ('is_patch', {'dtype': np.bool_, 'units': '1', 'description': 'True if element is a patch', 'default': False}),
        ('speed_factor', {'dtype': np.float32, 'units': '1', 'description': 'Base speed factor', 'default': 1.0}),
        ('target_lon', {'dtype': np.float32, 'units': 'deg', 'description': 'Target longitude'}),
        ('target_lat', {'dtype': np.float32, 'units': 'deg', 'description': 'Target latitude'}),
        ( 'target_patch_index', {'dtype': np.int32, 'units': '1', 'description': 'Index of target patch', 'default': -1}),
        ('collected_value',{'dtype': np.float32, 'units': '1', 'description': 'Total value collected by the boat', 'default': 0.0}),
        ('distance_traveled', {'dtype': np.float32, 'units': 'km', 'default': 0.0}),
    ])


class GeneticBoat(OpenDriftSimulation):
    ElementType = GeneticBoatArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class GeneticPlanner:
    def __init__(self, num_boats, num_patches, patches, current_data, patches_model):
        self.num_boats = num_boats
        self.num_patches = num_patches
        self.patches = patches
        self.current_data = current_data
        self.patches_model = patches_model
        self.start_position = (-133.5, 24.5)

    def evaluate(self, boat_histories):  # die strecke steht stellvertretend auch für energieverbrauch,sie soll also auch minimiert werdn, jedoch nur minimal in die fitness eingehen
        total_reward = 0.0
        total_distance = 0.0
        for hist in boat_histories:
            if len(hist) == 0:
                continue
            final = hist[-1]
            total_reward += float(final['collected_value'])
            total_distance += float(final['distance_traveled'])
        return 2.0 * total_reward - 0.1 * total_distance  # trade-off mit Distanz  # Beispiel für trade-off


    def generate_greedy_solutions(self, alphas):
        greedy_solutions = []
        for alpha in alphas:
            solution = self.run_greedy_solver(alpha)
            greedy_solutions.append(solution)
        return greedy_solutions

    def run_greedy_solver(self, alpha):
        sim = SimBoat(
            patches_model=self.patches_model.clone(),
            target_mode="weighted",
            weighted_alpha=alpha
        )

        for i in range(self.num_boats):
            sim.seed_boat(lon=self.start_position[0], lat=self.start_position[1])

        sim.run(duration=timedelta(hours=12))
        return sim.get_structured_history()



    """

    def simulate_route_opendrift(self, route):
        sim = SimBoat(
            patches_model=self.patches_model.clone(),
            target_mode="manual"
        )

        sim.seed_boat(lon=self.start_position[0], lat=self.start_position[1])
        boat_idx = 0

        for patch_id in route:
            patch = self.patches[patch_id]
            sim.elements.target_patch_index[boat_idx] = patch_id
            sim.elements.target_lon[boat_idx] = patch['lon']
            sim.elements.target_lat[boat_idx] = patch['lat']

            sim.run(duration=timedelta(hours=1))  # Simuliere Schrittweise

        sim.run(duration=timedelta(hours=1))  # Absicherung, falls Ziel nicht ganz erreicht
        history = sim.get_structured_history()
        reward = float(sim.elements.collected_value[boat_idx])
        time_spent = float(sim.elements.age_seconds[boat_idx]) / 3600.0
        return reward, time_spent



    def crossover(self, parent1, parent2):
        child = []
        for route1, route2 in zip(parent1, parent2):
            if random.random() < 0.5:
                child.append(route1.copy())
            else:
                child.append(route2.copy())
        return child

    def mutate(self, individual, mutation_rate=0.1):
        for route in individual:
            if len(route) > 1 and random.random() < mutation_rate:
                i, j = random.sample(range(len(route)), 2)
                route[i], route[j] = route[j], route[i]

    def select_parents(self, population, fitness_scores):
        idx1, idx2 = random.sample(range(len(population)), 2)
        return (population[idx1] if fitness_scores[idx1] > fitness_scores[idx2] else population[idx2]), \
            (population[idx2] if fitness_scores[idx2] > fitness_scores[idx1] else population[idx1])

    def run(self, pop_size=50, generations=100, use_greedy_alphas=True):
        greedy_solutions = []
        if use_greedy_alphas:
            alphas = [0.0, 0.3, 0.5, 0.7, 1.0]
            greedy_solutions = self.generate_greedy_solutions(alphas)

        population = self.initialize_population(pop_size, greedy_solutions)
        for generation in range(generations):
            fitness_scores = [self.evaluate(ind) for ind in population]
            new_population = []
            for _ in range(pop_size):
                parent1, parent2 = self.select_parents(population, fitness_scores)
                child = self.crossover(parent1, parent2)
                self.mutate(child)
                new_population.append(child)
            population = new_population
        best_individual = max(population, key=self.evaluate)
        return best_individual
    """