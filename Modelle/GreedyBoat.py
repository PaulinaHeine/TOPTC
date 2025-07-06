import numpy as np
import logging
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
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
import math
from scipy.spatial import cKDTree
matplotlib.use('Qt5Agg')

# === Logger gezielt stummschalten ===
logger = logging.getLogger(__name__)
# Schalte den allgemeinen OpenDrift-Logger auf "nur Fehler anzeigen"
logging.getLogger('opendrift').setLevel(logging.ERROR)

# Schalte den Logger deines eigenen Plastik-Modells ebenfalls stumm
# Der Name hier ('Modelle.OpenDriftPlastCustom') muss dem Pfad deiner Datei entsprechen.
logging.getLogger('Modelle.OpenDriftPlastCustom').setLevel(logging.ERROR)

# Die globale Konfiguration kann als zusätzlicher Fallback bleiben
logging.basicConfig(level=logging.ERROR)






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
    def __init__(self, patches_model,

                 # Grundstrategie
                 weighted_alpha=0.5,
                 adaptive_alpha_mode=True,


                 # Parameter für Opportunismus
                 enable_retargeting=True,
                 retarget_threshold=1.1,
                 opportunistic_alpha=0.9,
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
        self.initial_weighted_alpha = weighted_alpha

        # Parameter für adaptive Strategie speichern
        self.adaptive_alpha_enabled = adaptive_alpha_mode


        # NEU: Parameter für opportunistisches Re-Targeting
        self.retarget_threshold = retarget_threshold
        self.opportunistic_alpha = opportunistic_alpha
        self.enable_retargeting = enable_retargeting

    def update(self):
        """
        Führt einen Simulationsschritt aus und orchestriert die Entscheidungsfindung.
        """
        super().update()



        self.environment = self.get_environment(
            variables=['x_sea_water_velocity', 'y_sea_water_velocity'],
            time=self.time, lon=self.elements.lon, lat=self.elements.lat, z=self.elements.z, profiles = None
        )[0]

        for i in range(self.num_elements_active()):
            has_target = self.elements.target_patch_index[i] != -1
            if self.enable_retargeting and has_target:
                self._re_evaluate_target(i)
            elif not has_target:
                self.assign_target_weighted(i)

        self.move_toward_target()
        self.record_custom_history()
        self.elements.age_seconds += self.time_step.total_seconds()

    # =============================================================================
    # Basisfunktionen
    # =============================================================================

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


    # =============================================================================
    # Adapted alpha
    # =============================================================================

    def calculate_heuristic_alpha(self,
            time_frame,
            speed_factor_boat,
            enable_retargeting,
            boat_number,
            plastic_number,
            plastic_radius
    ):
        """
        Berechnet ein heuristisches Start-Alpha basierend auf Ihrem Regelwerk.
        Alpha nahe 0.2 = Explorer (Wert-Fokus), Alpha nahe 0.8 = Collector (Distanz-Fokus).
        """
        # 1. DEFINIERE ERWARTETE WERTEBEREICHE (Zum Anpassen)
        param_ranges = {
            'time_range': (50, 1000),
            'speed_range': (1, 5),
            'boat_range': (1, 10),
            'density_range': (0.1, 50.0),
            'area_range': (math.pi * 5 ** 2, math.pi * 50 ** 2)
        }

        # 2. GEWICHTUNG DER FAKTOREN (höher = wichtiger)
        weights = {
            'time': 0.50,  # von 0.80 reduziert
            'speed': 0.25,  # unverändert
            'area': 0.20,  # unverändert
            'density': 0.40,  # von 0.15 erhöht
            'boats': 0.40,  # von 0.15 erhöht
        }

        # 3. NORMALISIERTE SCORES BERECHNEN (0 bis 1)
        time_score = np.interp(time_frame, param_ranges['time_range'], [0, 1])
        speed_score = np.interp(speed_factor_boat, param_ranges['speed_range'], [0, 1])
        boat_score = np.interp(boat_number, param_ranges['boat_range'], [0, 1])

        area = np.pi * plastic_radius ** 2 if plastic_radius > 0 else 1
        density = plastic_number / area
        density_score = np.interp(density, param_ranges['density_range'], [0, 1])
        area_score = np.interp(area, param_ranges['area_range'], [0, 1])

        # 4. BERECHNE DEN "COLLECTOR-DRUCK" (0 = reiner Explorer, 1 = reiner Collector)
        # Ein hoher Score bedeutet, dass die Faktoren das Alpha nach oben (Richtung 0.8, Distanzfokus) treiben.

        # Wenig Zeit, niedrige Geschwindigkeit, wenig Boote und geringe Dichte erhöhen den Collector-Druck.
        collector_tendency = (
                                     weights['time'] * (1 - time_score) +
                                     weights['speed'] * (1 - speed_score) +
                                     weights['boats'] * (1 - boat_score) +
                                     weights['density'] * (1 - density_score)+
                                     weights['area'] *   ( area_score)
                             ) / sum(weights.values())

        base_tendency = collector_tendency

        # 5. FAKTOR "ENABLE_RETARGETING" ANWENDEN
        # Wenn True, sinkt der Collector-Druck (Alpha wird niedriger -> mehr Explorer),
        # da lange Wege zum "Exploren" attraktiver werden.
        if enable_retargeting:
            collector_tendency *= 0.7  # Reduziert den Collector-Druck (verschiebt Alpha Richtung 0.2)

        # 6. SKALIERE DAS ERGEBNIS AUF DEN ZIELBEREICH [0.2, 0.8]
        final_alpha = 0.2+(0.8 - 0.2) * collector_tendency
        final_alpha = np.clip(final_alpha, 0.2, 0.8)

        # --- Transparente Ausgabe der Entscheidung ---
        print("\n--- 🧠 Heuristik für Alpha (v3 nach Ihrer Definition) ---")
        print(
            f"  - Faktoren für EXPLORER (niedriges Alpha): Viel Zeit ({time_score:.2f}), hohe Geschwindigkeit ({speed_score:.2f}), viele Boote ({boat_score:.2f}), hohe Dichte ({density_score:.2f})")
        print(f"  - Faktoren für COLLECTOR (hohes Alpha): Wenig Zeit, langsam, wenig Boote, geringe Dichte")
        print(f"  - Berechnung:")
        print(f"    - Basis 'Collector-Tendenz' (0=Explorer, 1=Collector): {base_tendency:.2f}")
        if enable_retargeting:
            print(f"    - Modifiziert durch Retargeting: {collector_tendency:.2f}")
        print(f"==> Finales Alpha: {final_alpha:.2f} (nahe 0.2 = Explorer, nahe 0.8 = Collector)")
        print("---------------------------------------------------\n")

        return final_alpha


    def calculate_time_adjusted_alpha(self, time_frame: int, enable_retargeting: bool) :
        """
        Berechnet Alpha basierend auf einem festen Startwert von 0.6,
        der nur durch die Zeit justiert wird.

        Args:
            time_frame: Die Missionsdauer in Stunden.

        Returns:
            Ein Alpha-Wert zwischen 0.4 und 0.8.
        """
        # 1. Definiere die Fixpunkte
        base_alpha = 0.6
        min_alpha = 0.4
        max_alpha = 0.8

        # Definiere, was als "kurze" und "lange" Mission gilt
        # Kann bei Bedarf angepasst werden
        time_range = (50, 800)  # 50h = kürzeste Mission, 1000h = längste

        # 2. Berechne den Zeitausschlag
        # Normalisiere die Zeit auf einen Wert von 0 (kurz) bis 1 (lang)
        time_score = np.interp(time_frame, time_range, [0, 1])

        # Berechne den maximal möglichen Ausschlag vom Basiswert
        max_ausschlag_nach_oben = max_alpha - base_alpha  # -> +0.2
        max_ausschlag_nach_unten = min_alpha - base_alpha  # -> -0.4

        # Mappe den time_score auf den Ausschlag-Bereich
        # Kurze Zeit (score=0) -> maximal positiver Ausschlag (+0.2)
        # Lange Zeit (score=1) -> maximal negativer Ausschlag (-0.4)
        adjustment = np.interp(time_score, [0, 1], [max_ausschlag_nach_oben, max_ausschlag_nach_unten])


        # 3. Berechne das finale Alpha
        final_alpha = base_alpha + adjustment


        # 4. Wende den festen Retargeting-Abzug an
        if enable_retargeting:
            retargeting_abzug = 0.2
            final_alpha -= retargeting_abzug

        # Sicherstellen, dass das Ergebnis exakt im erlaubten Bereich liegt
        final_alpha = np.clip(final_alpha, min_alpha, max_alpha)

        print(f"\n--- Heuristik (Base: {base_alpha}, Zeit: {time_frame}h) ---")
        print(f"  - Zeitausschlag: {adjustment:+.2f}")
        print(f"==> Finales Alpha: {final_alpha:.2f}")

        return final_alpha




    # =============================================================================
    # Re_retargettinga
    # =============================================================================

    def _re_evaluate_target(self, boat_idx):
        """
        Prüft für ein Boot, das bereits ein Ziel hat, ob es eine deutlich bessere Gelegenheit gibt.
        """
        current_target_id = self.elements.target_patch_index[boat_idx]
        if current_target_id == -1: return

        # --- DEBUG START ---
        #print(f"\n--- Boot {boat_idx} bei Stunde {self.time.hour} prüft Ziel {current_target_id} ---")
        # --- DEBUG ENDE ---

        # 1. Score des aktuellen Ziels mit dem normalen Alpha berechnen
        _, current_score = self._find_best_candidate_patch(boat_idx, alpha_override=None)
        if current_score == -1:
            #print(f"DEBUG: Konnte Score für aktuelles Ziel {current_target_id} nicht berechnen.")
            return

        # 2. Beste Alternative mit dem opportunistischen (hohen) Alpha finden
        best_alternative_id, best_alternative_score = self._find_best_candidate_patch(boat_idx,
                                                                                      alpha_override=self.opportunistic_alpha)
        '''
        # --- DEBUG START ---
        #print(f"DEBUG: Score aktuelles Ziel ({current_target_id}): {current_score:.3f}")
        if best_alternative_id is not None:
            print(f"DEBUG: Beste Alternative ({best_alternative_id}): Score {best_alternative_score:.3f} (berechnet mit opportunistischem alpha={self.opportunistic_alpha})")
        else:
            print("DEBUG: Keine Alternativen gefunden.")
        # --- DEBUG ENDE ---
        '''

        # 3. Vergleichen und ggf. Kurs ändern
        if best_alternative_id is not None and best_alternative_id != current_target_id:
            required_score = current_score * self.retarget_threshold



            # --- DEBUG START ---
            #print(f"DEBUG: Vergleich: Ist {best_alternative_score:.3f} > {current_score:.3f} * {self.retarget_threshold} (also > {required_score:.3f})?")
            # --- DEBUG ENDE ---

            if best_alternative_score > required_score:
                # Diese Zeile solltest du jetzt sehen, wenn ein Wechsel stattfindet
                #print(f"----> ERFOLG! Boot {boat_idx} wechselt Ziel von {current_target_id} zu {best_alternative_id}!")
                self.elements.target_patch_index[boat_idx] = best_alternative_id
                self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[best_alternative_id]
                self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[best_alternative_id]
                self.elements.last_event_code[boat_idx] = 2 # wir wechseln das ziel
            # --- DEBUG START ---
            else:
                1+1 #was soll hier sonst hin, nur platzhalter
                #print("DEBUG: Nein, Bedingung nicht erfüllt. Bleibe auf Kurs.")
            # --- DEBUG ENDE ---


    # =============================================================================
    # History und Print funktionen
    # =============================================================================


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
        #print("\n🚤 Collection Summary:")
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

            #print(f"\n🚤 Boat {i}:")
            #print(f"   📦 Collected Value: {value:.2f}")
            #print(f"   📍 Distance Traveled: {distance:.2f} km")

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











def run_greedy(time_frame=100, plastic_radius=10, plastic_number=500, plastic_seed=1,
               boat_number=2, speed_factor_boat=3, animation=False,
               weighted_alpha_value=0.5, adaptive_alpha_mode=True,

               retarget_threshold=1.1, opportunistic_alpha=0.9, enable_retargeting=True):
    # Initiate


    # Logging konfigurieren
    logging.basicConfig(level=logging.WARNING)#info
    # Datenpfad
    data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024'
    # Dataset laden
    try:
        ds = xr.open_dataset(data_path)
        #print(ds)
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
                   adaptive_alpha_mode=adaptive_alpha_mode,
                   retarget_threshold=retarget_threshold,
                   opportunistic_alpha=opportunistic_alpha,
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
    #print("🚤 Speed-Faktoren nach Seeding:", b.elements.speed_factor[:b.num_elements_active()])


    if adaptive_alpha_mode==True:

        new_alpha = b.calculate_time_adjusted_alpha(time_frame, enable_retargeting)


            #speed_factor_boat,
            #enable_retargeting,
            #boat_number,
            #plastic_number,
            #plastic_radius)


        # Überschreibe den Wert für alle Elemente im Boot-Modell
        b.elements.weighted_alpha[:] = new_alpha
        print(f"Alpha-Wert wurde auf {new_alpha:.2f} überschrieben.")

    o.prepare_run()
    b.prepare_run()




    r = 0
    for i in range(0,steps):
        #print("Aktueller Simulationszeitpunkt:", o.time)
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


    if animation == True:
        try:
            animation_custom(model = o,fast = True, compare= b,size='value', show_trajectories=False)
            #b.plot(fast = True, show_trajectories=True,show_initial=True) # zeigt die routen der boote an
            #b.plot_custom(fast=True, linecolor='royalblue')
        except Exception as e:
            print(f"Fehler bei der Animation: {e}")

    if adaptive_alpha_mode == True:
        return short_logbook_for_graph, b.history, target_patch_route, new_alpha

    return short_logbook_for_graph, b.history, target_patch_route




if __name__ == "__main__":
   # Starte die Simulation mit der neuen, sich selbst kalibrierenden Strategie
   run_greedy(

         # === Umgebungs-Parameter ===
         time_frame=200,
         plastic_radius=10,
         plastic_number=100,
         plastic_seed=1,

         # === Boots-Parameter ===
         boat_number=2,
         speed_factor_boat=3,
         weighted_alpha_value = 0.0,

         # === Strategie-Parameter ===
         adaptive_alpha_mode = 0,

         # === Opportunismus-Parameter ===
         enable_retargeting=0,
         retarget_threshold= 1,
         opportunistic_alpha=0.95,

         # === Ausgabe-Parameter ===
         animation=0
    )


# je größer radius desto mehr dsitanz