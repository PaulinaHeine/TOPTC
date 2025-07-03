import numpy as np
import logging
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from datetime import datetime, timedelta
import matplotlib
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import weighted

from datetime import datetime, timedelta
import xarray as xr
from opendrift.readers.reader_netCDF_CF_generic import Reader
from visualisations.animations import animation_custom, plot_custom
from Modelle.OpenDriftPlastCustom import OpenDriftPlastCustom
from collections import defaultdict
import random
matplotlib.use('Qt5Agg')

logger = logging.getLogger(__name__)


class GreedyBoatArray(LagrangianArray):
    """
    Extends LagrangianArray with boat-specific variables.
    This array holds all properties for each boat or patch in the simulation.
    """
    variables = LagrangianArray.add_variables([
        ('speed_factor', {'dtype': np.float32, 'units': '1', 'description': 'Base speed factor', 'default': 1.0}),
        ('target_lon', {'dtype': np.float32, 'units': 'deg', 'description': 'Target longitude'}),
        ('target_lat', {'dtype': np.float32, 'units': 'deg', 'description': 'Target latitude'}),
        ('current_drift_factor', {'dtype': np.float32, 'units': '1', 'description': 'For compatibility', 'default': 10.0}),
        ('is_patch', {'dtype': np.bool_, 'units': '1', 'description': 'True if element is a patch', 'default': False}),
        ('target_patch_index', {'dtype': np.int32, 'units': '1', 'description': 'Index of target patch', 'default': -1}),
        ('collected_value', {'dtype': np.float32, 'units': '1', 'description': 'Total value collected by the boat', 'default': 0.0}),
        ('distance_traveled', {'dtype': np.float32, 'units': 'km', 'description': 'Total distance traveled', 'default': 0.0}),
        # NEU: Jedes Boot bekommt sein eigenes, potenziell dynamisches Alpha
        ('weighted_alpha', {'dtype': np.float32, 'units': '1','description': 'Weighting factor for target selection (0=Value, 1=Distance)','default': 0.5}),
        ('last_event_code', {'dtype': np.int8, 'units': '1', 'description': 'Code for the last event', 'default': 0}), # 0 nichts besonderes, 1 patch eingesammelt, 2 Ziel gewechselt
    ])


class GreedyBoat(OpenDriftSimulation):
    """
    Simulates a boat that greedily collects waste patches.
    The targeting strategy can be static (fixed alpha) or adaptive based on local patch density.
    """
    ElementType = GreedyBoatArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
    }

    # GEÄNDERT: __init__ akzeptiert jetzt den Schalter und die neuen Parameter
    def __init__(self, patches_model, weighted_alpha=0.5, adaptive_alpha=False,
                 scan_radius_km=15.0, min_density_for_alpha=1,   # Dichte, bei der Alpha minimal ist
                 max_density_for_alpha=20,  # Dichte, bei der Alpha maximal ist
                 retarget_threshold=1.2, opportunistic_alpha=0.9,enable_retargeting=True,
                 *args, **kwargs):

        """
        Initializes the GreedyBoat simulation model.

        Args:
            patches_model: An OpenDrift model instance that contains collectible patches.
            weighted_alpha (float): The default or static weighting factor.
            adaptive_alpha (bool): If True, alpha is adjusted dynamically.
            scan_radius_km (float): Radius in km for density check.
            density_threshold (int): Number of patches to be considered a "dense" area.
            high_density_alpha (float): Alpha value for dense areas (Collector-Mode).
            low_density_alpha (float): Alpha value for sparse areas (Explorer-Mode).
        """
        super().__init__(*args, **kwargs)
        self.patches_model = patches_model
        self.adaptive_alpha_enabled = adaptive_alpha

        # Parameter für adaptive Strategie
        self.scan_radius_km = scan_radius_km
        self.min_density = min_density_for_alpha
        self.max_density = max_density_for_alpha
        self.initial_weighted_alpha = weighted_alpha

        # NEU: Parameter für opportunistisches Re-Targeting
        self.retarget_threshold = retarget_threshold
        self.opportunistic_alpha = opportunistic_alpha
        self.enable_retargeting = enable_retargeting

    def update(self):
        """
        Performs one simulation step.
        """
        super().update()
        self.environment = self.get_environment(
            variables=['x_sea_water_velocity', 'y_sea_water_velocity'],
            time=self.time, lon=self.elements.lon, lat=self.elements.lat, z=self.elements.z,
            profiles=None
        )[0]

        if self.adaptive_alpha_enabled:
            self._update_adaptive_alpha()

        # GEÄNDERTE LOGIK mit Schalter
        for i in range(self.num_elements_active()):
            has_target = self.elements.target_patch_index[i] != -1

            # Fall 1: Re-Targeting ist AN und das Boot hat bereits ein Ziel
            if self.enable_retargeting and has_target:
                self._re_evaluate_target(i)
            # Fall 2: Das Boot hat KEIN Ziel (dieser Fall wird immer ausgeführt, wenn nötig)
            elif not has_target:
                self.assign_target_weighted(i)

        self.move_toward_target()
        self.record_custom_history()
        self.elements.age_seconds += self.time_step.total_seconds()



    # NEU: Hilfsfunktion zum Scannen der Umgebung
    def _get_local_patch_density(self, boat_idx):
        """
        Zählt die Anzahl der aktiven Patches innerhalb des Scan-Radius um ein Boot.
        """
        count = 0
        boat_lon, boat_lat = self.elements.lon[boat_idx], self.elements.lat[boat_idx]
        radius_deg = self.scan_radius_km / 111.0

        # Nur aktive Patches im patches_model berücksichtigen
        active_patch_indices = np.where(self.patches_model.elements.status == 0)[0]

        for i in active_patch_indices:
            patch_lon = self.patches_model.elements.lon[i]
            patch_lat = self.patches_model.elements.lat[i]
            dist = np.sqrt((patch_lon - boat_lon) ** 2 + (patch_lat - boat_lat) ** 2)
            if dist < radius_deg:
                count += 1
        return count

    def _update_adaptive_alpha(self):
        """
        Passt den Alpha-Wert kontinuierlich basierend auf der lokalen Patch-Dichte an.
        """
        for i in range(self.num_elements_active()):
            # 1. Lokale Dichte wie bisher ermitteln
            local_density = self._get_local_patch_density(i)

            # 2. Den Anteil innerhalb unseres definierten Dichte-Bereichs berechnen
            # Sicherstellen, dass der Nenner nicht Null ist
            density_range = self.max_density - self.min_density
            if density_range <= 0:
                density_fraction = 0.5  # Standardwert, falls min/max falsch gesetzt sind
            else:
                density_fraction = (local_density - self.min_density) / density_range

            # 3. Den Anteil auf den Bereich 0.0 bis 1.0 begrenzen
            density_fraction = np.clip(density_fraction, 0.0, 1.0)

            # 4. Den neuen Alpha-Wert zuweisen
            # Wir könnten hier noch einen min/max Alpha-Wert definieren, aber für 0 bis 1 ist es einfach der Anteil.
            new_alpha = density_fraction
            self.elements.weighted_alpha[i] = new_alpha

    def check_and_pick_new_target(self, threshold_km=0.1):
        """
        Checks if any active boat needs a new target (i.e., its current target_patch_index is -1)
        and assigns one using the weighted target selection strategy.

        Args:
            threshold_km (float): A distance threshold in kilometers. Currently not directly used
                                  in this specific function, but could be for future logic.
        """
        for i in range(self.num_elements_active()):
            if self.elements.target_patch_index[i] == -1:
                self.assign_target_weighted(i)

    def move_toward_target(self):
        """
        Calculates and applies the movement for each active boat towards its assigned target.
        This includes calculating step size, normalizing direction, and applying current drift influence.
        If a boat reaches its target patch, the patch is collected (deactivated) and a new target is assigned.
        """
        for i in range(self.num_elements_active()):
            patch_idx = int(self.elements.target_patch_index[i])
            if patch_idx < 0 or patch_idx >= self.patches_model.num_elements_total():
                continue
            if np.isnan(self.patches_model.elements.lat[patch_idx]) or np.isnan(
                    self.patches_model.elements.lon[patch_idx]):
                self.elements.target_patch_index[i] = -1
                continue

            target_lon = self.patches_model.elements.lon[patch_idx]
            target_lat = self.patches_model.elements.lat[patch_idx]

            dlon = target_lon - self.elements.lon[i]
            dlat = target_lat - self.elements.lat[i]
            dist = np.sqrt(dlon ** 2 + dlat ** 2)

            if dist < (0.1 / 111.0):  # Check if boat is within collection radius (approx. 0.1km)
                self.deactivate_patch_near(self.elements.lat[i], self.elements.lon[i], boat_idx=i)
                self.elements.target_patch_index[i] = -1
                self.elements.last_event_code[i] = 1 # wir haben etwas eingesammelt
                self.assign_target_weighted(i)
                continue

            dlon_norm = dlon / dist
            dlat_norm = dlat / dist

            max_step_deg = (0.06 / 111.0) * self.elements.speed_factor[i] # Base speed of ~0.06 km/timestep
            step_deg = min(dist, max_step_deg)
            step_lon = dlon_norm * step_deg
            step_lat = dlat_norm * step_deg

            try:
                u = self.environment['x_sea_water_velocity'][i]
                v = self.environment['y_sea_water_velocity'][i]
            except Exception as e:
                logger.warning(f"⚠️ No current data for boat {i}: {e}")
                u, v = 0.0, 0.0

            drift_scale = 3600 / 111000 # Convert m/s to degrees per hour for current influence
            drift_lon = u * drift_scale
            drift_lat = v * drift_scale

            drift_influence = 0.1 # Factor to reduce direct current influence
            step_lon += drift_influence * drift_lon
            step_lat += drift_influence * drift_lat

            self.elements.lon[i] += step_lon
            self.elements.lat[i] += step_lat

            # Calculate and add distance traveled in km
            self.elements.distance_traveled[i] += np.sqrt(step_lon ** 2 + step_lat ** 2) * 111.0



    def deactivate_patch_near(self, lat, lon, boat_idx=None, radius_km=0.100):
        """
        Deactivates (removes) patches that are within a specified radial distance
        from a given latitude and longitude.
        If `boat_idx` is provided, the value of the deactivated patch is added to that boat's
        `collected_value`.

        Args:
            lat (float): Latitude of the center point for the radius search.
            lon (float): Longitude of the center point for the radius search.
            boat_idx (int, optional): The index of the boat that collected the patch.
                                      If None, the patch is simply deactivated without value transfer.
            radius_km (float): The radius in kilometers within which patches will be deactivated.
        """
        threshold_deg = radius_km / 111.0 # Convert km to degrees for comparison.
        num = self.patches_model.num_elements_total()

        for i in range(num):
            # Skip if already inactive, not a patch, or outside the radius.
            if self.patches_model.elements.status[i] != 0:
                continue
            if not self.patches_model.elements.is_patch[i]:
                continue

            d = np.sqrt(
                (self.patches_model.elements.lat[i] - lat) ** 2 +
                (self.patches_model.elements.lon[i] - lon) ** 2
            )

            if d < threshold_deg:
                if boat_idx is not None:
                    self.elements.collected_value[boat_idx] += self.patches_model.elements.value[i]

                # Mark patch as collected/removed by setting value to 0 and position to NaN.
                self.patches_model.elements.value[i] = 0.0
                self.patches_model.elements.lat[i] = np.nan
                self.patches_model.elements.lon[i] = np.nan

        # Officially deactivate elements in the patch model whose positions are NaN.
        self.patches_model.deactivate_elements(np.isnan(self.patches_model.elements.lat))


    def seed_boat(self, lon, lat, number=1, time=None, speed_factor=1.0):
        """
        Seeds new boat elements into the simulation.
        """
        self.seed_elements(
            lon=lon, lat=lat, number=number, time=time,
            target_lon=lon, target_lat=lat, speed_factor=speed_factor,
            # NEU: Setze das Alpha auf den initialen Wert
            weighted_alpha=self.initial_weighted_alpha
        )
        self.release_elements()

    def validate_boats(self):
        """
        Performs a diagnostic check on the state of the boats.
        Logs issues such as boats with NaN positions, no assigned target, or being stuck (speed 0, no target).
        """
        total = self.num_elements_total()
        active = self.num_elements_active()

        stuck = 0
        nan_pos = 0
        no_target = 0

        for i in range(active):
            lat = self.elements.lat[i]
            lon = self.elements.lon[i]
            tgt = self.elements.target_patch_index[i]
            speed = self.elements.speed_factor[i]

            if np.isnan(lat) or np.isnan(lon):
                nan_pos += 1
            if tgt == -1:
                no_target += 1
            if speed == 0.0 and tgt == -1: # Condition for a boat being stuck.
                stuck += 1

        logger.info(f"Total={total}, Active={active}, NaN-pos={nan_pos}, No-target={no_target}, Stuck={stuck}")

    def record_custom_history(self):
        """
        Records a snapshot of specific boat variables for each boat at the current timestep.
        This data is stored internally and can be retrieved later for detailed historical analysis.
        """
        if not hasattr(self, 'custom_history_list'):
            self.custom_history_list = []

        num_elements = self.num_elements_total()
        for i in range(num_elements):
            # Create a tuple of relevant data for the current boat.
            entry = (
                int(i),
                self.elements.status[i],
                self.elements.moving[i],
                float(self.elements.age_seconds[i]),
                i,
                float(self.elements.lon[i]),
                float(self.elements.lat[i]),
                float(self.elements.speed_factor[i]),
                float(self.elements.target_lon[i]),
                float(self.elements.target_lat[i]),
                float(self.elements.collected_value[i]),
                float(self.elements.distance_traveled[i]),
                int(self.elements.target_patch_index[i]),
                float(self.elements.weighted_alpha[i]),
                int(self.elements.last_event_code[i])
            )

            # Ensure a list exists for this boat's history.
            while len(self.custom_history_list) <= i:
                self.custom_history_list.append([])
            # Append the current entry to the boat's history.
            self.custom_history_list[i].append(entry)

            if self.elements.last_event_code[i] != 0:
                self.elements.last_event_code[i] = 0

    def get_structured_history(self):
        """
        Converts the internally stored custom history data into a structured NumPy masked array.
        This format is highly efficient for data analysis and plotting, allowing easy access
        to all recorded variables over time for each boat.

        Returns:
            numpy.ma.MaskedArray: A structured masked array containing the historical data.
        """
        import numpy.ma as ma

        # Define the data type (schema) for the structured array.
        dtype = [
            ('ID', 'i4'), ('status', 'i4'), ('moving', 'i4'), ('age', 'f8'), ('idx', 'i4'),
            ('lon', 'f8'), ('lat', 'f8'), ('speed_factor', 'f4'),
            ('target_lon', 'f8'), ('target_lat', 'f8'), ('collected_value', 'f4'),
            ('distance_traveled', 'f4'), ('target_patch_index', 'i4'),('weighted_alpha', 'f4'),('last_event_code', 'i1')
        ]

        all_records = []
        # Convert each boat's history list to a structured masked array.
        for boat in self.custom_history_list:
            arr = np.array(boat, dtype=dtype)
            all_records.append(ma.masked_array(arr))

        # Stack all individual boat arrays into one cohesive structure.
        return ma.stack(all_records)

    def print_collection_summary(self):
        """
        Prints a summary of the total value collected and total distance traveled
        by each active boat, along with overall totals.
        Also returns a simplified logbook of these values.

        Returns:
            list: A list of lists, where each inner list contains [collected_value, distance_traveled]
                  for each boat, with the last entry being the total values.
        """
        print("\n🚤 Collection Summary:")
        total_value = 0
        total_distance = 0
        total_collected = 0 # Count of boats that collected at least one item.
        short_logbook = []

        for i in range(self.num_elements_active()):
            value = float(self.elements.collected_value[i])
            distance = float(self.elements.distance_traveled[i])
            short_logbook.append([value, distance])
            if value > 0:
                total_collected += 1
            total_value += value
            total_distance += distance

            print(f"\n🚤 Boat {i}:")
            print(f"   📦 Collected Value: {value:.2f}")
            print(f"   📍 Distance Traveled: {distance:.2f} km")

        print(f"\n📦 Total Value: {total_value:.2f}")
        print(f"🚗 Total Distance: {total_distance:.2f} km")
        print(f"✅ Boats with Collection: {total_collected} of {self.num_elements_active()}")
        short_logbook.append([total_value, total_distance]) # Add overall totals to the logbook.

        return short_logbook

    def extract_routes_from_history(self, history_data, boat_number):
        solutions = [[] for _ in range(boat_number)]
        last_target = [-1] * boat_number

        for boat_idx in range(boat_number):
            # Wir gehen die Timesteps für jedes Boot durch
            if boat_idx < len(history_data):
                for timestep_data in history_data[boat_idx]:
                    # Das letzte Element ist die target_patch_index
                    current_target = timestep_data[-1]

                    # Wenn sich das Ziel geändert hat und das alte Ziel nicht -1 war,
                    # wurde ein Patch erreicht und ein neues Ziel gesetzt.
                    if current_target != last_target[boat_idx] and last_target[boat_idx] != -1:
                        # Füge das *alte* (gerade erreichte) Ziel zur Route hinzu
                        solutions[boat_idx].append(last_target[boat_idx])

                    last_target[boat_idx] = current_target

        return solutions

    def print_alpha_summary(self):
        """
        Gibt eine textbasierte Zusammenfassung der Alpha-Wert-Änderungen für jedes Boot aus.
        """
        # Greift direkt auf die im Objekt gespeicherte Historie zu
        if not hasattr(self, 'history') or self.history is None:
            print("Keine History-Daten für die Alpha-Zusammenfassung vorhanden.")
            return

        boat_history = self.history
        num_boats = boat_history.shape[0]
        if num_boats == 0:
            print("Keine Bootsdaten für die Alpha-Zusammenfassung vorhanden.")
            return

        print("\n--- 🧠 Alpha-Strategie Zusammenfassung ---")

        for i in range(num_boats):
            print(f"Boot {i}:")

            valid_steps = not boat_history[i]['age'].mask.all()
            if not valid_steps:
                print("  - Keine gültigen Daten für dieses Boot.")
                continue

            age = boat_history[i]['age'][~boat_history[i]['age'].mask]
            alpha = boat_history[i]['weighted_alpha'][~boat_history[i]['weighted_alpha'].mask]

            if len(alpha) == 0:
                continue

            last_alpha = alpha[0]
            print(f"  - Start-Alpha: {last_alpha:.2f}")

            for t_step in range(1, len(alpha)):
                if alpha[t_step] != last_alpha:
                    time_in_hours = age[t_step] / 3600
                    print(f"  - Wechsel zu {alpha[t_step]:.2f} bei Stunde {time_in_hours:.1f}")
                    last_alpha = alpha[t_step]

    def _find_best_candidate_patch(self, boat_idx, alpha_override=None):
        """
        Sucht nach dem besten verfügbaren Patch für ein Boot und gibt dessen Index und Score zurück.
        Kann mit einem speziellen Alpha-Wert aufgerufen werden.
        """
        boat_lon, boat_lat = self.elements.lon[boat_idx], self.elements.lat[boat_idx]
        taken_targets = set(self.elements.target_patch_index[:self.num_elements_active()])

        candidates = []
        active_patch_indices = np.where(self.patches_model.elements.status == 0)[0]

        for i in active_patch_indices:
            is_own_target = (i == self.elements.target_patch_index[boat_idx])
            if not self.patches_model.elements.is_patch[i] or (i in taken_targets and not is_own_target):
                continue

            patch_value = self.patches_model.elements.value[i]
            dist = np.sqrt((self.patches_model.elements.lon[i] - boat_lon) ** 2 + (
                        self.patches_model.elements.lat[i] - boat_lat) ** 2)
            candidates.append({'id': i, 'dist': dist, 'value': patch_value})

        if not candidates:
            return None, -1

        max_dist = max(c['dist'] for c in candidates) or 1.0
        max_value = max(c['value'] for c in candidates) or 1.0

        current_alpha = alpha_override if alpha_override is not None else self.elements.weighted_alpha[boat_idx]

        best_candidate_id, max_score = None, -1
        for cand in candidates:
            norm_dist = 1 - cand['dist'] / max_dist
            norm_value = cand['value'] / max_value
            score = current_alpha * norm_dist + (1 - current_alpha) * norm_value
            if score > max_score:
                max_score = score
                best_candidate_id = cand['id']

        return best_candidate_id, max_score

    def assign_target_weighted(self, boat_idx):
        """
        Weist einem Boot ohne Ziel das beste verfügbare Ziel zu.
        """
        best_patch_id, _ = self._find_best_candidate_patch(boat_idx)
        if best_patch_id is not None:
            self.elements.target_patch_index[boat_idx] = best_patch_id
            self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[best_patch_id]
            self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[best_patch_id]

    def _re_evaluate_target(self, boat_idx):
        """
        Prüft für ein Boot, das bereits ein Ziel hat, ob es eine deutlich bessere Gelegenheit gibt.
        """
        current_target_id = self.elements.target_patch_index[boat_idx]
        if current_target_id == -1: return

        # --- DEBUG START ---
        print(f"\n--- Boot {boat_idx} bei Stunde {self.time.hour} prüft Ziel {current_target_id} ---")
        # --- DEBUG ENDE ---

        # 1. Score des aktuellen Ziels mit dem normalen Alpha berechnen
        _, current_score = self._find_best_candidate_patch(boat_idx, alpha_override=None)
        if current_score == -1:
            print(f"DEBUG: Konnte Score für aktuelles Ziel {current_target_id} nicht berechnen.")
            return

        # 2. Beste Alternative mit dem opportunistischen (hohen) Alpha finden
        best_alternative_id, best_alternative_score = self._find_best_candidate_patch(boat_idx,
                                                                                      alpha_override=self.opportunistic_alpha)

        # --- DEBUG START ---
        print(f"DEBUG: Score aktuelles Ziel ({current_target_id}): {current_score:.3f}")
        if best_alternative_id is not None:
            print(
                f"DEBUG: Beste Alternative ({best_alternative_id}): Score {best_alternative_score:.3f} (berechnet mit opportunistischem alpha={self.opportunistic_alpha})")
        else:
            print("DEBUG: Keine Alternativen gefunden.")
        # --- DEBUG ENDE ---

        # 3. Vergleichen und ggf. Kurs ändern
        if best_alternative_id is not None and best_alternative_id != current_target_id:
            required_score = current_score * self.retarget_threshold

            # --- DEBUG START ---
            print(
                f"DEBUG: Vergleich: Ist {best_alternative_score:.3f} > {current_score:.3f} * {self.retarget_threshold} (also > {required_score:.3f})?")
            # --- DEBUG ENDE ---

            if best_alternative_score > required_score:
                # Diese Zeile solltest du jetzt sehen, wenn ein Wechsel stattfindet
                print(f"----> ERFOLG! Boot {boat_idx} wechselt Ziel von {current_target_id} zu {best_alternative_id}!")
                self.elements.target_patch_index[boat_idx] = best_alternative_id
                self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[best_alternative_id]
                self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[best_alternative_id]
                self.elements.last_event_code[boat_idx] = 2 # wir wechseln das ziel
            # --- DEBUG START ---
            else:
                print("DEBUG: Nein, Bedingung nicht erfüllt. Bleibe auf Kurs.")
            # --- DEBUG ENDE ---

        # In der Klasse GreedyBoat...




def run_greedy(time_frame=100, plastic_radius=10, plastic_number=500, plastic_seed=1,
               boat_number=2, speed_factor_boat=3, animation=False,
               weighted_alpha_value=0.5, adaptive_alpha_mode=False,
               scan_radius_km=15.0,                min_density_for_alpha=1,
               max_density_for_alpha=20,retarget_threshold=1.2, opportunistic_alpha=0.9, enable_retargeting=True):
    # Initiate

    # Logging konfigurieren
    logging.basicConfig(level=logging.WARNING)#info
    # Datenpfad
    data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024'
    # Dataset laden
    try:
        ds = xr.open_dataset(data_path)
        print(ds)
    except FileNotFoundError:
        print(f"Fehler: Die Datei unter '{data_path}' wurde nicht gefunden.")
        print("Bitte stellen Sie sicher, dass der 'data_path' korrekt ist und die Datei existiert.")
        return
    except Exception as e:
        print(f"Fehler beim Laden des Datasets: {e}")
        return

    # Plastikmodell initialisieren
    o = OpenDriftPlastCustom(loglevel=logging.INFO)
    r = Reader(data_path)
    o.add_reader(r)



    # boot initalisieren
    b = GreedyBoat(loglevel=logging.INFO,
                   patches_model=o,
                   weighted_alpha=weighted_alpha_value,
                   adaptive_alpha=adaptive_alpha_mode,

                   scan_radius_km=scan_radius_km,
                   min_density_for_alpha=min_density_for_alpha,
                   max_density_for_alpha=max_density_for_alpha,
                    retarget_threshold = retarget_threshold,
                    opportunistic_alpha = opportunistic_alpha,
                   enable_retargeting=enable_retargeting
    )
    b.add_reader(r)



    lon_min = float(ds.longitude.min())
    lon_max = float(ds.longitude.max())
    lat_min = float(ds.latitude.min())
    lat_max = float(ds.latitude.max())

    o.simulation_extent = [lon_min, lon_max, lat_min, lat_max]
    b.simulation_extent = [lon_min, lon_max, lat_min, lat_max]

    # Startzeit vorbereiten
    start_time = ds.time.values[0]
    if not isinstance(start_time, datetime):
        start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)

    # Geografisches Zentrum des Gebiets
    mid_latitude = ds.latitude[int(len(ds.latitude) / 2)]
    mid_longitude = ds.longitude[int(len(ds.longitude) / 2)]
    depth = ds.depth.values[0]


    steps = time_frame
    dt = timedelta(hours=1)
    o.time_step = dt
    o.time_step_output = timedelta(hours=1)
    o.time = start_time

    b.time_step = dt
    b.time_step_output = timedelta(hours=1)
    b.time = start_time

    o.seed_plastic_patch(radius_km = plastic_radius,number = plastic_number, lon=mid_longitude, lat=mid_latitude, time = start_time, z = depth, seed = plastic_seed)
    b.seed_boat(lon=mid_longitude, lat=mid_latitude,number=boat_number, time = start_time, speed_factor= speed_factor_boat) #ca 6kmh
    print("🚤 Speed-Faktoren nach Seeding:", b.elements.speed_factor[:b.num_elements_active()])

    o.prepare_run()
    b.prepare_run()



    r = 0
    for i in range(0,steps):
        print("Aktueller Simulationszeitpunkt:", o.time)
        b.update()
        o.update()
        r += 1

        # 4. Zeit voranschreiten
        o.time += dt
        b.time += dt
        i += 1




    o.history = o.get_structured_history()


    b.history = b.get_structured_history()

    short_logbook_for_graph = b.print_collection_summary()
    #print(b.history)
    target_patch_route = b.extract_routes_from_history(b.history, boat_number)
    #print(target_patch_route)

    if adaptive_alpha_mode:
        b.print_alpha_summary()
    """
    if animation == True:
        try:
            # Rufe nur noch die eine, neue Funktion auf, die alles erledigt.
            create_and_plot_decisions(b)
        except Exception as e:
            print(f"Fehler bei der Visualisierung: {e}")
    """
    if animation == True:
        try:
            animation_custom(model = o,fast = True, compare= b,size='value', show_trajectories=False)
            #b.plot(fast = True, show_trajectories=True,show_initial=True) # zeigt die routen der boote an
            #b.plot_custom(fast=True, linecolor='royalblue')
        except Exception as e:
            print(f"Fehler bei der Animation: {e}")

    return short_logbook_for_graph, b.history, target_patch_route

if __name__ == "__main__":
   # Starte die Simulation mit einer spezifischen Konfiguration
   run_greedy(
         # === Umgebungs-Parameter ===
         time_frame = 200,              # Legt die Dauer der Simulation auf 100 Stunden fest.
         plastic_radius = 10,              # Erzeugt die Plastik-Patches in einem engen Radius von 3 km.
         plastic_number = 1100,               # Definiert die Anzahl der erzeugten Plastik-Patches.
         plastic_seed = 1,                  # Sorgt für eine reproduzierbare, zufällige Verteilung der Patches.

         # === Boots-Parameter ===
         boat_number = 1,                   # Setzt die Anzahl der Boote in der Simulation.
         speed_factor_boat = 1,             # Das Boot fährt mit seiner normalen Basis-Geschwindigkeit.

         # === Strategie-Parameter ===
         weighted_alpha_value = 0.5,        # Die Hauptstrategie des Bootes: 100% Fokus auf den Wert eines Ziels (0.0), ignoriert die Distanz. value 0 > dist 1
         adaptive_alpha_mode = False,       # Die Strategie ist statisch und ändert sich nicht automatisch je nach Dichte.

         # Die folgenden zwei Werte sind für diesen Lauf inaktiv, da adaptive_alpha_mode=False ist.
         scan_radius_km = 0.8,              # Radius, in dem das Boot die Patch-Dichte prüfen würde. 2 /10 des radius?
        min_density_for_alpha=1,                #Eine Untergrenze (z.B. 1 Patch), bei der das Boot im reinen "Explorer-Modus" ist (Alpha nahe 0, Fokus auf Wert).
        max_density_for_alpha=8,            # Eine Obergrenze (z.B. 20 Patches), bei der das Boot im reinen "Collector-Modus" ist (Alpha nahe 1, Fokus auf Distanz).

         # === Opportunismus-Parameter ===
         enable_retargeting = True,         # Das Boot darf von seinem Hauptziel abweichen, wenn sich eine gute Gelegenheit bietet.
         retarget_threshold = 0.99999,          # Schwelle für den Kurswechsel: Eine neue Gelegenheit muss einen Score von >80% des aktuellen Ziels erreichen.
         opportunistic_alpha = 0.9,         # Bei der Suche nach Gelegenheiten fokussiert sich das Boot zu 80% auf die Distanz.

         # === Ausgabe-Parameter ===
         animation = True               # Zeigt am Ende eine grafische Animation der Simulation.
    )
