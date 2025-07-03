import random
import numpy as np
import time
import copy
from datetime import datetime, timedelta

# Importiere die notwendigen Klassen und Funktionen aus deinen Dateien
from GreedyBoat import GreedyBoat, run_greedy
from OpenDriftPlastCustom import OpenDriftPlastCustom
from opendrift.readers.reader_netCDF_CF_generic import Reader
import logging
import xarray as xr


# --- Hinzufügen, um OpenDrift-Logs zu verbergen ---
# Hole den spezifischen Logger für die "opendrift" Bibliothek
od_logger = logging.getLogger('opendrift')
# Setze das Level auf ERROR, sodass INFO und WARNING nicht mehr angezeigt werden
od_logger.setLevel(logging.ERROR)
# ------------------------------------------------

# ---------------------------------------------------------------------------
# NEUE BOOTS-KLASSE, DIE NUR PLÄNE AUSFÜHRT
# ---------------------------------------------------------------------------
class RouteFollowingBoat(GreedyBoat):
    """
    Diese Klasse erbt von GreedyBoat, aber ihre Update-Methode wurde so
    verändert, dass sie keine eigenen Entscheidungen zur Zielauswahl mehr trifft.
    Sie ist ein reiner "Befehlsempfänger" und folgt stur dem vorgegebenen Plan.
    """

    def update(self):
        # Führe die wichtige Basis-Logik von OpenDrift aus (z.B. Alterung)
        super(GreedyBoat, self).update()

        # Hole die Umgebungsinformationen
        self.environment = self.get_environment(
            variables=['x_sea_water_velocity', 'y_sea_water_velocity'],
            time=self.time,
            lon=self.elements.lon,
            lat=self.elements.lat,
            z=self.elements.z,
            profiles=None
        )[0]

        # WICHTIG: Wir lassen diesen Teil, das "Gehirn" des Bootes, absichtlich weg
        # self.check_and_pick_new_target()

        # Führe nur die Bewegung zum vorgegebenen Ziel und das Logging aus
        self.move_toward_target()
        self.record_custom_history()


# ---------------------------------------------------------------------------
# 1. HELFERKLASSE ZUR BEWERTUNG EINER LÖSUNG (FITNESSFUNKTION)
#
# ---------------------------------------------------------------------------

class SimulationEvaluator:
    def __init__(self, simulation_params):
        self.params = simulation_params
        self.base_patches_model = self._create_base_patches_model()

    def _create_base_patches_model(self):
        try:
            ds = xr.open_dataset(self.params['data_path'])
        except Exception as e:
            print(f"Fehler beim Laden der Daten mit xarray: {e}")
            raise

        patches_model = OpenDriftPlastCustom(loglevel=logging.WARNING)
        reader_instance = Reader(self.params['data_path'])
        patches_model.add_reader(reader_instance)

        patches_model.simulation_extent = [
            float(ds.longitude.min()), float(ds.longitude.max()),
            float(ds.latitude.min()), float(ds.latitude.max())
        ]

        start_time = ds.time.values[0]
        if not isinstance(start_time, datetime):
            start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)
        patches_model.time = start_time

        dt = timedelta(hours=1)
        patches_model.time_step = dt
        patches_model.time_step_output = dt

        mid_lat = ds.latitude[int(len(ds.latitude) / 2)]
        mid_lon = ds.longitude[int(len(ds.longitude) / 2)]

        patches_model.seed_plastic_patch(
            radius_km=self.params['plastic_radius'],
            number=self.params['plastic_number'],
            lon=mid_lon, lat=mid_lat,
            time=start_time,
            seed=self.params['plastic_seed']
        )

        patches_model.prepare_run()

        drift_hours = self.params.get('drift_pre_steps', 0) # todo prüfen was die 0 hier macht und ob ich das in dem greedy auch reinbringe
        for _ in range(drift_hours):
            patches_model.update()
            patches_model.time += dt

        return patches_model

    def evaluate_solution(self, solution, max_time_hours):
        try:
            patches_model_instance = copy.deepcopy(self.base_patches_model)
        except Exception as e:
            print(f"FATAL: deepcopy ist fehlgeschlagen: {e}")
            return -1, max_time_hours + 1, -1

        num_boats = len(solution)
        if num_boats == 0:
            return 0, 0, 0

        # ### KORREKTUR: Wir benutzen jetzt unsere spezialisierte RouteFollowingBoat ###
        boat_model = RouteFollowingBoat(loglevel=logging.ERROR, patches_model=patches_model_instance)
        boat_model.add_reader(Reader(self.params['data_path']))
        boat_model.simulation_extent = patches_model_instance.simulation_extent
        boat_model.time = patches_model_instance.time
        dt = timedelta(hours=1)
        boat_model.time_step = dt
        boat_model.time_step_output = dt

        # ### KORREKTUR: Sorge für exakt dieselbe Startposition wie in run_greedy ###
        # Berechne die geographische Mitte aus der simulation_extent
        lon_min, lon_max, lat_min, lat_max = boat_model.simulation_extent
        mid_lon = lon_min + (lon_max - lon_min) / 2
        mid_lat = lat_min + (lat_max - lat_min) / 2

        boat_model.seed_boat(
            lon=mid_lon, lat=mid_lat, number=num_boats,  # <-- Start in der Mitte
            time=boat_model.time, speed_factor=self.params['speed_factor_boat']
        )
        boat_model.prepare_run()

        # Ab hier bleibt der Rest der Funktion identisch.
        # Sie führt den Plan jetzt unter den korrekten Startbedingungen aus.
        routes_to_follow = [list(r) for r in solution]
        for i in range(num_boats):
            boat_model.elements.target_patch_index[i] = -1
            if routes_to_follow[i]:
                while routes_to_follow[i]:
                    next_target_id = routes_to_follow[i].pop(0)
                    target_idx = next_target_id - 1
                    if 0 <= target_idx < patches_model_instance.num_elements_total() and \
                            patches_model_instance.elements.status[target_idx] == 0:
                        boat_model.elements.target_patch_index[i] = target_idx
                        break

        step = 0
        for step in range(max_time_hours):
            boat_model.update()
            patches_model_instance.update()
            boat_model.time += dt
            patches_model_instance.time += dt

            all_boats_finished = True
            for i in range(num_boats):
                if routes_to_follow[i] or boat_model.elements.target_patch_index[i] != -1:
                    all_boats_finished = False
                    break
            if all_boats_finished: break

            for i in range(num_boats):
                if boat_model.elements.target_patch_index[i] == -1 and routes_to_follow[i]:
                    while routes_to_follow[i]:
                        next_target_id = routes_to_follow[i].pop(0)
                        target_idx = next_target_id - 1
                        if 0 <= target_idx < patches_model_instance.num_elements_total() and \
                                patches_model_instance.elements.status[target_idx] == 0:
                            boat_model.elements.target_patch_index[i] = target_idx
                            break

        total_collected_value = np.sum(boat_model.elements.collected_value)
        total_distance_km = np.sum(boat_model.elements.distance_traveled)
        return total_collected_value, step + 1, total_distance_km


# ---------------------------------------------------------------------------
# 2. GENETISCHE OPERATOREN
#
# ---------------------------------------------------------------------------

def tournament_selection(population, fitnesses, k=3):
    selected_indices = random.sample(range(len(population)), k)
    best_index_in_tournament = -1
    best_fitness = -1
    for i in selected_indices:
        if fitnesses[i] > best_fitness:
            best_fitness = fitnesses[i]
            best_index_in_tournament = i
    return population[best_index_in_tournament]


def order_crossover(parent1, parent2):
    size = len(parent1)
    child = [-1] * size
    start, end = sorted(random.sample(range(size), 2))
    child[start:end] = parent1[start:end]
    p2_idx = end
    c_idx = end
    while -1 in child:
        if parent2[p2_idx % size] not in child:
            child[c_idx % size] = parent2[p2_idx % size]
            c_idx += 1
        p2_idx += 1
    return child



def multi_boat_crossover(parent1_solution, parent2_solution):
    """
    Führt Crossover für Multi-Boot-Lösungen durch und ist robust gegen
    leere oder ungleich lange Lösungen.
    """
    # 1. Mache aus den Elternteilen "Riesen-Touren"
    giant_tour1 = [patch for route in parent1_solution for patch in route]
    giant_tour2 = [patch for route in parent2_solution for patch in route]

    # Sicherheitsabfrage 1: Prüfe auf leere Touren
    if not giant_tour1 or not giant_tour2:
        return parent1_solution, parent2_solution

    # --- NEUE, ENTSCHEIDENDE SICHERHEITSABFRAGE ---
    # Unser 'order_crossover' funktioniert nur, wenn beide Eltern
    # dieselbe Anzahl an Gesamt-Patches haben (gleiche Länge der Riesen-Tour).
    if len(giant_tour1) != len(giant_tour2):
        # Wenn nicht, ist Crossover schwierig. Wir geben die Eltern
        # einfach zurück und hoffen auf die nächste Paarung oder Mutation.
        return parent1_solution, parent2_solution
    # -----------------------------------------------

    # Behalte die ursprünglichen Routenlängen für die spätere Aufteilung
    route_lengths1 = [len(r) for r in parent1_solution]
    route_lengths2 = [len(r) for r in parent2_solution]

    # 2. Kreuze die Riesen-Touren mit dem bekannten Crossover
    child1_giant_tour = order_crossover(giant_tour1, giant_tour2)
    child2_giant_tour = order_crossover(giant_tour2, giant_tour1)

    # 3. Teile die neuen Riesen-Touren wieder auf die Boote auf
    child1_solution, idx = [], 0
    for length in route_lengths1:
        child1_solution.append(child1_giant_tour[idx: idx + length])
        idx += length

    child2_solution, idx = [], 0
    for length in route_lengths2:
        child2_solution.append(child2_giant_tour[idx: idx + length])
        idx += length

    return child1_solution, child2_solution


def mutate_solution(solution, mutation_rate=0.05):
    if random.random() < mutation_rate:
        # Füge alle Routen zu einer zusammen, tausche zwei Patches und teile sie wieder auf
        route_lengths = [len(r) for r in solution]
        giant_tour = [patch for route in solution for patch in route]

        if len(giant_tour) >= 2:
            idx1, idx2 = random.sample(range(len(giant_tour)), 2)
            giant_tour[idx1], giant_tour[idx2] = giant_tour[idx2], giant_tour[idx1]

        # Teile die mutierte Tour wieder auf
        mutated_solution, idx = [], 0
        for length in route_lengths:
            mutated_solution.append(giant_tour[idx: idx + length])
            idx += length
        return mutated_solution
    return solution


# ---------------------------------------------------------------------------
# 3. DIE HAUPTFUNKTION DES GENETISCHEN ALGORITHMUS
# ---------------------------------------------------------------------------

def run_genetic_algorithm(initial_population, simulation_params, max_time_hours, generations=50, population_size=20,
                          mutation_rate=0.05, elitism_size=2):
    print("🚀 Starte Genetischen Algorithmus...")
    evaluator = SimulationEvaluator(simulation_params)
    population = initial_population
    best_solution_overall = None
    best_fitness_overall = -1
    # ### NEU: Speichere auch die beste Distanz ###
    best_distance_overall = -1

    for gen in range(generations):
        start_time_gen = time.time()
        print(f"\n--- Generation {gen + 1}/{generations} ---")

        fitnesses = []
        distances = []  # ### NEU: Liste für die Distanzen ###
        for i, solution in enumerate(population):
            print(f"  Bewerte Individuum {i + 1}/{len(population)}...", end='\r')

            # ### GEÄNDERT: Erhalte jetzt 3 Rückgabewerte ###
            value, t, distance = evaluator.evaluate_solution(solution, max_time_hours)

            fitnesses.append(value)
            distances.append(distance)  # ### NEU ###

        print("\n  Evaluation abgeschlossen.")

        max_fitness_gen = max(fitnesses) if fitnesses else 0
        best_index_gen = fitnesses.index(max_fitness_gen) if fitnesses else -1

        if max_fitness_gen > best_fitness_overall:
            best_fitness_overall = max_fitness_gen
            best_solution_overall = population[best_index_gen]
            # ### NEU: Speichere die Distanz der besten Lösung ###
            best_distance_overall = distances[best_index_gen]
            print(
                f"  🎉 Neuer bester Wert gefunden: {best_fitness_overall:.2f} (Distanz: {best_distance_overall:.2f} km)")

        new_population = []
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
        new_population.extend(sorted_population[:elitism_size])

        while len(new_population) < population_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # ### GEÄNDERT ###: Benutze den neuen Crossover
            child1, child2 = multi_boat_crossover(parent1, parent2)

            # ### GEÄNDERT ###: Benutze die neue Mutation
            child1 = mutate_solution(child1, mutation_rate)
            child2 = mutate_solution(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population
        gen_duration = time.time() - start_time_gen
        print(f"  Bester Wert in Gen. {gen + 1}: {max_fitness_gen:.2f}")
        print(f"  Dauer der Generation: {gen_duration:.2f}s")

    print("\n✅ Genetischer Algorithmus abgeschlossen!")
    # ### GEÄNDERT: Gebe auch die beste Distanz zurück ###
    return best_solution_overall, best_fitness_overall, best_distance_overall


# ---------------------------------------------------------------------------
# 4. SKRIPT-AUSFÜHRUNG
#    ### GEÄNDERT ###: Erzeugt eine Startpopulation für mehrere Boote
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # 1. Parameter für die Simulation und den GA definieren
    simulation_params = {
        'data_path': '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024',
        'plastic_radius': 10,
        'plastic_number': 1000,
        'plastic_seed': 1,
        'speed_factor_boat': 3,
    }

    NUM_BOATS = 2
    MAX_TIME_HOURS = 100
    GENERATIONS = 10
    POPULATION_SIZE = 10
    MUTATION_RATE = 0.1
    ELITISM_SIZE = 2

    # 2. Startpopulation mit deinen Greedy-Lösungen erzeugen
    print(f"--- Erzeuge Startpopulation für {NUM_BOATS} Boote mit Greedy-Algorithmus ---")
    initial_population = []
    alphas_for_seeding = [0.0, 0.25, 0.5, 0.75, 1.0]

    for alpha in alphas_for_seeding:
        print(f"  Führe Greedy mit alpha = {alpha} aus...")

        _, _, greedy_solution = run_greedy(
            time_frame=MAX_TIME_HOURS,
            boat_number=NUM_BOATS,
            weighted_alpha_value=alpha,
            animation=False,
            plastic_radius=simulation_params['plastic_radius'],
            plastic_number=simulation_params['plastic_number'],
            plastic_seed=simulation_params['plastic_seed'],
            speed_factor_boat=simulation_params['speed_factor_boat']
        )
        initial_population.append(greedy_solution)

    print("\n--- Fülle restliche Population mit zufälligen Lösungen auf ---")
    all_patches = list(range(1, simulation_params['plastic_number'] + 1))
    while len(initial_population) < POPULATION_SIZE:
        random.shuffle(all_patches)
        random_solution = np.array_split(all_patches, NUM_BOATS)
        initial_population.append([list(route) for route in random_solution])

    print(f"\nStartpopulation mit {len(initial_population)} Individuen erzeugt.")

    # --- NEUER TEIL: Ausgabe der Startlösungen und ihrer Fitness ---
    print("\n--- Fitness der initialen Population ---")

    # Wir brauchen eine Instanz des Evaluators, um die Fitness zu berechnen
    # Wir setzen drift_pre_steps hier, da es nur für den GA relevant ist, nicht für die Greedy-Läufe
    ga_sim_params = {**simulation_params, 'drift_pre_steps': 0}
    evaluator = SimulationEvaluator(ga_sim_params)

    initial_fitnesses = []
    initial_distances = []  # ### NEU ###
    for i, solution in enumerate(initial_population):
        # ### GEÄNDERT: Erhalte 3 Rückgabewerte ###
        value, t, distance = evaluator.evaluate_solution(solution, MAX_TIME_HOURS)
        initial_fitnesses.append(value)
        initial_distances.append(distance)  # ### NEU ###

        solution_type = f"Greedy (alpha={alphas_for_seeding[i]})" if i < len(alphas_for_seeding) else "Zufällig"
        print(f"\n  Lösung {i + 1} ({solution_type}):")
        # ### GEÄNDERT: Gib auch die Distanz aus ###
        print(f"    -> Fitness: {value:.2f}, Distanz: {distance:.2f} km, Zeit: {t}h")
        for boat_idx, route in enumerate(solution):
            print(f"       Boot {boat_idx + 1} Route: {route}")
    # ----------------------------------------------------------------

    # Starte den Genetischen Algorithmus
    start_time_ga = time.time()
    best_solution, best_value, best_distance = run_genetic_algorithm(
        initial_population=initial_population,
        simulation_params=ga_sim_params,  # Wichtig: die GA-Parameter hier übergeben
        max_time_hours=MAX_TIME_HOURS,
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        elitism_size=ELITISM_SIZE
    )
    total_duration_ga = time.time() - start_time_ga

    # Präsentiere das Endergebnis
    print("\n\n--- FINALES ERGEBNIS ---")
    print(f"Bester initialer Wert: {max(initial_fitnesses):.2f}")
    print(f"Bester gefundener Wert nach {GENERATIONS} Generationen: {best_value:.2f} (Distanz: {best_distance:.2f} km)")
    print(f"Optimierte Routenaufteilung:")
    if best_solution:
        for i, route in enumerate(best_solution):
            print(f"  Boot {i+1}: {route}")
    print(f"\nGesamtdauer des Algorithmus: {total_duration_ga / 60:.2f} Minuten")

    # irgendwas läuft aber ich weiß nicht was oder wie, er nimmt als starlösungen irgendwei auch nicht die greedy lösungen--- daher wird auch die evaluation function nicht passen

    # FITNESS FUNCTION PRÜFEN!!!!!!



    #er sagt fitness ist 0 aber die werte simmen nicht!