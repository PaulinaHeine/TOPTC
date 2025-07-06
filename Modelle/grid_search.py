import itertools
import csv
from datetime import datetime

# Importiere deine Hauptfunktionen
# Ich habe den Import für die Heuristik-Funktion explizit gemacht.
try:
    from Modelle.GreedyBoat import *
except ImportError as e:
    print(f"FEHLER: Der Import ist fehlgeschlagen: {e}")
    exit()

def run_full_comparison_grid_search():
    """
    Führt einen Grid Search aus, der die Heuristik-Strategie mit
    verschiedenen statischen Alpha-Werten über diverse Szenarien vergleicht.
    """
    ## 1. Parameter-Raster definieren
    # ----------------------------------------------------------------------
    scenario_grid = {
        'time_frame': [200, 400],
        'plastic_number': [100,500],
        'plastic_radius': [10],
        'boat_number': [1,3],
        'speed_factor_boat': [1,3],
    }
    retargeting_on_grid = {
        'enable_retargeting': [True],
        'retarget_threshold': [0.999, 1.0, 1.1],
        'opportunistic_alpha': [0.9, 0.95],
    }
    retargeting_off_grid = {
        'enable_retargeting': [False],
    }
    static_alpha_values = [0.4, 0.5, 0.6,0.7,0.8]
    fixed_params = {'plastic_seed': 1000, 'animation': False}


    ## 2. Alle Test-Konfigurationen generieren
    # ----------------------------------------------------------------------
    def generate_configs(grid):
        keys, values = zip(*grid.items())
        return [dict(zip(keys, v)) for v in itertools.product(*values)]

    scenarios = generate_configs(scenario_grid)
    retargeting_options = generate_configs(retargeting_on_grid) + generate_configs(retargeting_off_grid)
    param_combinations = []

    # Block 1: Heuristik-Konfigurationen
    for scenario in scenarios:
        for retarget_option in retargeting_options:
            param_combinations.append({**scenario, **retarget_option, 'adaptive_alpha_mode': True})

    # Block 2: Statische Alpha-Konfigurationen
    for scenario in scenarios:
        for retarget_option in retargeting_options:
            for alpha in static_alpha_values:
                param_combinations.append({**scenario, **retarget_option, 'adaptive_alpha_mode': False, 'weighted_alpha_value': alpha})

    print(f"Starte Grid Search mit insgesamt {len(param_combinations)} Konfigurationen...")


    ## 3. Experiment durchführen und Ergebnisse speichern
    # ----------------------------------------------------------------------
    results_filename = f'grid_search_results_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
    fieldnames = ['adaptive_alpha_mode', 'weighted_alpha_value', 'heuristic_alpha']
    fieldnames += sorted(list(scenario_grid.keys()) + list(retargeting_on_grid.keys()))
    fieldnames += ['efficiency', 'total_value', 'total_distance']

    with open(results_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for i, params in enumerate(param_combinations):
            print(f"\n--- [{i + 1}/{len(param_combinations)}] Starte Lauf mit: {params} ---")
            current_config = {**fixed_params, **params}
            result_row = {**params}

            try:


                # Simulation ausführen

                if current_config['adaptive_alpha_mode'] is True:
                    logbook, _, _, heuristic_alpha = run_greedy(**current_config)

                else:
                    logbook, _, _ = run_greedy(**current_config)

                # Ergebnisse auswerten
                result_row['heuristic_alpha'] = heuristic_alpha
                total_value = logbook[-1][0]
                total_distance = logbook[-1][1]
                efficiency = total_value / total_distance if total_distance > 0 else 0

                print(f"-> Ergebnis: Wert={total_value:.2f}, Distanz={total_distance:.2f}, Effizienz={efficiency:.4f}")

                # Ergebnis in CSV schreiben
                result_row.update({'efficiency': efficiency, 'total_value': total_value, 'total_distance': total_distance})
                writer.writerow(result_row)

            except Exception as e:
                print(f"!!!! Lauf fehlgeschlagen mit Konfiguration {params}: {e}")

        print(f"\n\nGrid Search beendet. Ergebnisse in '{results_filename}' gespeichert.")

if __name__ == "__main__":
    run_full_comparison_grid_search()