import random
import numpy as np
import time
from datetime import datetime, timedelta
import copy

import numpy as np
import logging
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from datetime import datetime, timedelta
import matplotlib
from scipy.cluster.hierarchy import weighted
from opendrift.models.basemodel import OpenDriftSimulation
from datetime import datetime, timedelta
import xarray as xr
from opendrift.readers.reader_netCDF_CF_generic import Reader
from visualisations.animations import animation_custom
from Modelle.OpenDriftPlastCustom import OpenDriftPlastCustom
from collections import defaultdict
import random
matplotlib.use('Qt5Agg')

# Importiere die notwendigen Klassen und Funktionen aus deinen Dateien
from GreedyBoat import GreedyBoat, run_greedy
from OpenDriftPlastCustom import OpenDriftPlastCustom
from opendrift.readers.reader_netCDF_CF_generic import Reader
import logging

# Reduziere das Logging, um die Konsolenausgabe sauber zu halten
logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# 1. HELFERKLASSE ZUR BEWERTUNG EINER LÖSUNG (FITNESSFUNKTION)
# ---------------------------------------------------------------------------

class SimulationEvaluator:
    """
    Diese Klasse ist dafür verantwortlich, eine einzelne Lösung (eine Route)
    zu nehmen und ihre Fitness zu berechnen, indem sie eine Simulation ausführt.
    Sie initialisiert die Simulationsumgebung einmal, um sie für viele
    Evaluationen wiederzuverwenden.
    """

    def __init__(self, simulation_params):
        """
        Initialisiert die Basis-Simulationsumgebung.
        """
        self.params = simulation_params
        self.base_patches_model = self._create_base_patches_model()



    # In der Klasse SimulationEvaluator:
    def _create_base_patches_model(self):
        """
        Erstellt ein initiales, unberührtes Patch-Modell, das als Vorlage
        für jede Simulation dient.
        """
        # 1. Lade das Dataset separat mit xarray, um Grenzen etc. zu erhalten
        try:
            ds = xr.open_dataset(self.params['data_path'])
        except Exception as e:
            print(f"Fehler beim Laden der Daten mit xarray: {e}")
            raise

        # 2. Erstelle und konfiguriere das Plastik-Modell
        patches_model = OpenDriftPlastCustom(loglevel=logging.WARNING)

        # 3. Erstelle eine Instanz des Readers und füge sie dem Modell hinzu
        reader_instance = Reader(self.params['data_path'])
        patches_model.add_reader(reader_instance)

        # 4. Verwende das 'ds'-Objekt, um die Grenzen zu definieren
        patches_model.simulation_extent = [
            float(ds.longitude.min()), float(ds.longitude.max()),
            float(ds.latitude.min()), float(ds.latitude.max())
        ]

        start_time = ds.time.values[0]
        if not isinstance(start_time, datetime):
            start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)
        patches_model.time = start_time

        # Setze die Dauer für einen Zeitschritt auf eine Stunde
        dt = timedelta(hours=1)
        patches_model.time_step = dt
        patches_model.time_step_output = dt

        # 5. Seede die Patches deterministisch
        mid_lat = ds.latitude[int(len(ds.latitude) / 2)]
        mid_lon = ds.longitude[int(len(ds.longitude) / 2)]

        patches_model.seed_plastic_patch(
            radius_km=self.params['plastic_radius'],
            number=self.params['plastic_number'],
            lon=mid_lon, lat=mid_lat,
            time=start_time,
            seed=self.params['plastic_seed']
        )

        # --------- WICHTIGE ÄNDERUNG DES ABLAUFS ---------
        # 6. Finalisiere das Seeding mit prepare_run(), wie im Original-Code
        patches_model.prepare_run()

        # 7. Simuliere die Vordrift mit einer manuellen Schleife statt mit .run()
        drift_hours = self.params.get('drift_pre_steps', 10)
        for _ in range(drift_hours):
            patches_model.update()
            patches_model.time += dt  # Manuell die Zeit fortschreiben
        # ----------------------------------------------------

        return patches_model

    def evaluate_route(self, route, max_time_hours):
        """
        Simuliert eine einzelne Route und gibt den gesammelten Wert und die Zeit zurück.
        """
        try:
            patches_model_instance = copy.deepcopy(self.base_patches_model)
        except Exception as e:
            print(f"FATAL: deepcopy ist fehlgeschlagen: {e}")
            return -1, max_time_hours + 1

        # Erstelle und konfiguriere das Boot-Modell
        boat_model = GreedyBoat(loglevel=logging.WARNING, patches_model=patches_model_instance)
        boat_model.add_reader(Reader(self.params['data_path']))
        boat_model.time = patches_model_instance.time
        boat_model.simulation_extent = patches_model_instance.simulation_extent

        # --- HIER IST DIE FEHLENDE KORREKTUR ---
        # Auch das Boot-Modell benötigt seinen Zeitschritt.
        boat_model.time_step = timedelta(hours=1)
        boat_model.time_step_output = timedelta(hours=1)
        # ----------------------------------------

        # Seede das Boot
        boat_lon = patches_model_instance.elements.lon[0]
        boat_lat = patches_model_instance.elements.lat[0]
        boat_model.seed_boat(
            lon=boat_lon, lat=boat_lat, number=1,
            time=boat_model.time, speed_factor=self.params['speed_factor_boat']
        )
        boat_model.prepare_run()

        # Konvertiere Route zu Indizes
        try:
            route_indices = [pid - 1 for pid in route if 0 <= pid - 1 < patches_model_instance.num_elements_total()]
        except Exception:
            return 0, max_time_hours + 1

        time_steps = 0

        # Weise das erste gültige Ziel aus der Route zu
        if route_indices:
            while route_indices:
                next_target = route_indices.pop(0)
                if patches_model_instance.elements.status[next_target] == 0:
                    boat_model.elements.target_patch_index[0] = next_target
                    break
                # Wenn alle Ziele aus der Route schon weg sind, startet das Boot ohne Ziel
                if not route_indices:
                    boat_model.elements.target_patch_index[0] = -1

        # Simulations-Schleife
        for step in range(max_time_hours):
            if boat_model.elements.target_patch_index[0] == -1:
                break  # Boot hat kein Ziel mehr

            boat_model.update()
            patches_model_instance.update()

            boat_model.time += timedelta(hours=1)
            patches_model_instance.time += timedelta(hours=1)
            time_steps += 1

            # Wenn Ziel erreicht, weise nächstes Ziel zu
            if boat_model.elements.target_patch_index[0] == -1:
                if route_indices:
                    while route_indices:
                        next_target = route_indices.pop(0)
                        if patches_model_instance.elements.status[next_target] == 0:
                            boat_model.elements.target_patch_index[0] = next_target
                            break
                else:
                    break  # Keine Ziele mehr in der Route

        collected_value = float(boat_model.elements.collected_value[0])
        return collected_value, time_steps


# ---------------------------------------------------------------------------
# 2. GENETISCHE OPERATOREN
# ---------------------------------------------------------------------------

def tournament_selection(population, fitnesses, k=3):
    """Wählt ein Individuum per Turnier-Selektion aus."""
    selected_indices = random.sample(range(len(population)), k)
    best_index_in_tournament = -1
    best_fitness = -1

    for i in selected_indices:
        if fitnesses[i] > best_fitness:
            best_fitness = fitnesses[i]
            best_index_in_tournament = i

    return population[best_index_in_tournament]


def order_crossover(parent1, parent2):
    """Führt einen Order Crossover (OX1) durch."""
    size = len(parent1)
    child1, child2 = [-1] * size, [-1] * size

    # Zufälligen Abschnitt auswählen
    start, end = sorted(random.sample(range(size), 2))

    # Abschnitt von Eltern auf Kinder kopieren
    child1[start:end] = parent1[start:end]
    child2[start:end] = parent2[start:end]

    # Fülle die restlichen Gene von den anderen Elternteilen auf
    # Für Kind 1 mit Genen von Elternteil 2
    p2_idx = end
    c1_idx = end
    while -1 in child1:
        if parent2[p2_idx % size] not in child1:
            child1[c1_idx % size] = parent2[p2_idx % size]
            c1_idx += 1
        p2_idx += 1

    # Für Kind 2 mit Genen von Elternteil 1
    p1_idx = end
    c2_idx = end
    while -1 in child2:
        if parent1[p1_idx % size] not in child2:
            child2[c2_idx % size] = parent1[p1_idx % size]
            c2_idx += 1
        p1_idx += 1

    return child1, child2


def swap_mutation(route, mutation_rate=0.05):
    """Tauscht mit einer gewissen Wahrscheinlichkeit zwei Gene in der Route."""
    if random.random() < mutation_rate:
        idx1, idx2 = random.sample(range(len(route)), 2)
        route[idx1], route[idx2] = route[idx2], route[idx1]
    return route


# ---------------------------------------------------------------------------
# 3. DIE HAUPTFUNKTION DES GENETISCHEN ALGORITHMUS
# ---------------------------------------------------------------------------

def run_genetic_algorithm(initial_population, simulation_params, max_time_hours, generations=50, population_size=20,
                          mutation_rate=0.05, elitism_size=2):
    """
    Führt den kompletten genetischen Algorithmus aus.
    """
    print("🚀 Starte Genetischen Algorithmus...")

    # Initialisiere den Evaluator, der die Simulationsumgebung enthält
    evaluator = SimulationEvaluator(simulation_params)

    population = initial_population
    best_solution_overall = None
    best_fitness_overall = -1

    for gen in range(generations):
        start_time_gen = time.time()
        print(f"\n--- Generation {gen + 1}/{generations} ---")

        # 1. Fitness-Evaluation
        fitnesses = []
        for i, route in enumerate(population):
            print(f"  Bewerte Individuum {i + 1}/{len(population)}...", end='\r')
            value, t = evaluator.evaluate_route(route, max_time_hours)

            if t > max_time_hours:
                fitnesses.append(0)  # Strafe für Zeitüberschreitung
            else:
                fitnesses.append(value)
        print("\n  Evaluation abgeschlossen.")

        # Finde die beste Lösung dieser Generation
        max_fitness_gen = max(fitnesses)
        best_index_gen = fitnesses.index(max_fitness_gen)

        if max_fitness_gen > best_fitness_overall:
            best_fitness_overall = max_fitness_gen
            best_solution_overall = population[best_index_gen]
            print(f"  🎉 Neuer bester Wert gefunden: {best_fitness_overall:.2f}")

        # 2. Selektion und nächste Generation erstellen
        new_population = []

        # Elitismus: Die besten Individuen direkt übernehmen
        sorted_population = [x for _, x in sorted(zip(fitnesses, population), key=lambda pair: pair[0], reverse=True)]
        new_population.extend(sorted_population[:elitism_size])

        # Rest der Population mit Crossover und Mutation füllen
        while len(new_population) < population_size:
            # Eltern auswählen
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)

            # Kreuzung
            child1, child2 = order_crossover(parent1, parent2)

            # Mutation
            child1 = swap_mutation(child1, mutation_rate)
            child2 = swap_mutation(child2, mutation_rate)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population
        gen_duration = time.time() - start_time_gen
        print(f"  Bester Wert in Gen. {gen + 1}: {max_fitness_gen:.2f} (Gesamt: {best_fitness_overall:.2f})")
        print(f"  Dauer der Generation: {gen_duration:.2f}s")

    print("\n✅ Genetischer Algorithmus abgeschlossen!")
    return best_solution_overall, best_fitness_overall


# ---------------------------------------------------------------------------
# 4. SKRIPT-AUSFÜHRUNG
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Basis-Parameter für alle Simulationen
    simulation_params = {
        'data_path': '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024',
        'plastic_radius': 10,
        'plastic_number': 50,  # Kleinere Zahl für schnellere Tests
        'plastic_seed': 1,
        'boat_number': 2,  # Im GA bewerten wir immer nur eine Route für ein Boot
        'speed_factor_boat': 3,
        'drift_pre_steps': 0  # Patches 5h treiben lassen, bevor die Boote starten
    }

    # GA-Parameter
    MAX_TIME_HOURS = 100  # Zeitbudget für eine Route in Stunden
    GENERATIONS = 10
    POPULATION_SIZE = 10
    MUTATION_RATE = 0.1
    ELITISM_SIZE = 1

    # Schritt 1: Erzeuge eine diverse Startpopulation mit dem Greedy-Algorithmus
    print("--- Erzeuge Startpopulation mit Greedy-Algorithmus ---")
    initial_population = []
    # Verschiedene Alpha-Werte, um unterschiedliche Strategien (Wert vs. Distanz) zu erzeugen
    alphas_for_seeding = [0.0, 0.3, 0.6, 1.0]

    for alpha in alphas_for_seeding:
        print(f"  Führe Greedy mit alpha = {alpha} aus...")

        # HINWEIS: run_greedy müsste die gefahrene Route (Liste von Patch-IDs) zurückgeben.
        # Da dies nicht implementiert ist, simulieren wir hier eine plausible Route,
        # indem wir eine Liste aller Patch-IDs erstellen und mischen.
        # In der Praxis müsstest du `GreedyBoat` anpassen, um die Route aufzuzeichnen.

        # Behelfslösung: Generiere eine zufällige Route als Platzhalter
        # Alle möglichen Patch-IDs von 1 bis plastic_number
        placeholder_route = list(range(1, simulation_params['plastic_number'] + 1))
        random.shuffle(placeholder_route)
        initial_population.append(placeholder_route)

    # Fülle die restliche Population mit zufälligen Routen auf
    while len(initial_population) < POPULATION_SIZE:
        placeholder_route = list(range(1, simulation_params['plastic_number'] + 1))
        random.shuffle(placeholder_route)
        initial_population.append(placeholder_route)

    print(f"Startpopulation mit {len(initial_population)} Individuen erzeugt.")

    # Schritt 2: Führe den Genetischen Algorithmus aus
    start_time_ga = time.time()
    best_route, best_value = run_genetic_algorithm(
        initial_population=initial_population,
        simulation_params=simulation_params,
        max_time_hours=MAX_TIME_HOURS,
        generations=GENERATIONS,
        population_size=POPULATION_SIZE,
        mutation_rate=MUTATION_RATE,
        elitism_size=ELITISM_SIZE
    )
    total_duration_ga = time.time() - start_time_ga

    # Schritt 3: Präsentiere das Ergebnis
    print("\n\n--- FINALES ERGEBNIS ---")
    print(f"Bester gefundener Wert: {best_value:.2f}")
    print(f"Optimierte Route (Reihenfolge der Patch-IDs):")
    # Zeige die ersten 20 Schritte der besten Route
    print(f"  {best_route[:20]}...")
    print(f"\nGesamtdauer des Algorithmus: {total_duration_ga / 60:.2f} Minuten")