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
    ])


class GreedyBoat(OpenDriftSimulation):
    """
    Simulates a boat that greedily collects waste patches
    using a weighted strategy combining patch value and distance.
    """
    ElementType = GreedyBoatArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
    }

    def __init__(self, patches_model, weighted_alpha=0.5, *args, **kwargs):
        """
        Initializes the GreedyBoat simulation model.

        Args:
            patches_model: An OpenDrift model instance that contains the collectible patches.
            weighted_alpha (float): A weighting factor (0-1) for target selection.
                                    A higher value prioritizes distance (boats go to closer targets),
                                    a lower value prioritizes value (boats go to more valuable targets).
        """
        super().__init__(*args, **kwargs)
        self.patches_model = patches_model
        self.weighted_alpha = weighted_alpha

    def update(self):
        """
        Performs one simulation step.
        This method is called at each timestep by OpenDrift to update the boat's state.
        It fetches environmental data, assigns/checks targets, moves the boats, and records their history.
        """
        super().update()
        self.environment = self.get_environment(
            variables=['x_sea_water_velocity', 'y_sea_water_velocity'],
            time=self.time,
            lon=self.elements.lon,
            lat=self.elements.lat,
            z=self.elements.z,
            profiles=None
        )[0]
        self.check_and_pick_new_target()
        self.move_toward_target()
        self.record_custom_history()

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

    def assign_target_weighted(self, boat_idx):
        """
        Assigns the best available target patch to a specific boat.
        The "best" patch is determined by a weighted score that balances
        the patch's value and its distance from the boat, using `self.weighted_alpha`.

        Args:
            boat_idx (int): The index of the boat for which to assign a new target.
        """
        if self.patches_model.num_elements_active() == 0:
            return

        if self.elements.target_patch_index[boat_idx] != -1: # Already has a target
            return

        boat_lon = self.elements.lon[boat_idx]
        boat_lat = self.elements.lat[boat_idx]
        # Keep track of targets already assigned to other active boats to prevent duplicates.
        taken_targets = set(self.elements.target_patch_index[:self.num_elements_active()])

        candidates = []
        for i in range(self.patches_model.num_elements_total()):
            # Skip inactive, non-patch, or already targeted elements.
            if self.patches_model.elements.status[i] != 0:
                continue
            if not self.patches_model.elements.is_patch[i]:
                continue
            if i in taken_targets:
                continue

            patch_value = self.patches_model.elements.value[i]
            # Calculate Euclidean distance in degrees.
            dlon = self.patches_model.elements.lon[i] - boat_lon
            dlat = self.patches_model.elements.lat[i] - boat_lat
            dist = np.sqrt(dlon ** 2 + dlat ** 2)

            candidates.append((i, dist, patch_value))

        if not candidates: # No suitable patches found
            return

        # Normalize distance and value to a 0-1 range for scoring.
        max_dist = max([c[1] for c in candidates])
        max_value = max([c[2] for c in candidates]) or 1.0 # Prevent division by zero

        scored = []
        for i, dist, value in candidates:
            norm_dist = 1 - dist / max_dist  # Closer is better (higher score)
            norm_value = value / max_value   # Higher value is better (higher score)
            # Calculate weighted score.
            score = self.weighted_alpha * norm_dist + (1 - self.weighted_alpha) * norm_value
            scored.append((i, score))

        # Select the candidate with the highest score.
        best_idx, best_score = max(scored, key=lambda x: x[1])

        # Assign the selected patch as the new target for the boat.
        self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[best_idx]
        self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[best_idx]
        self.elements.target_patch_index[boat_idx] = best_idx

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
        Seeds new boat elements into the simulation at specified coordinates and time.

        Args:
            lon (float): Longitude for the new boat(s).
            lat (float): Latitude for the new boat(s).
            number (int): The number of boats to seed. Defaults to 1.
            time (datetime, optional): The specific time at which to seed the boats.
                                      Defaults to the current simulation time if not provided.
            speed_factor (float): The initial speed factor for the new boat(s).
        """
        pre_count = self.num_elements_total() # Store current total element count.

        # Use the base OpenDrift method to create new elements.
        self.seed_elements(
            lon=lon,
            lat=lat,
            number=number,
            time=time,
            target_lon=lon, # Initial target is self-position, will be updated.
            target_lat=lat,
            speed_factor=speed_factor
        )

        post_count = self.num_elements_total() # New total element count.
        newly_seeded = slice(pre_count, post_count) # Slice for newly added elements.

        # Ensure the speed factor is set for the new boats.
        self.elements.speed_factor[newly_seeded] = speed_factor
        self.release_elements() # Make the newly seeded elements active in the simulation.

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
            )

            # Ensure a list exists for this boat's history.
            while len(self.custom_history_list) <= i:
                self.custom_history_list.append([])
            # Append the current entry to the boat's history.
            self.custom_history_list[i].append(entry)

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
            ('distance_traveled', 'f4'), ('target_patch_index', 'i4')
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



def run_greedy( time_frame = 100, plastic_radius = 10, plastic_number = 500, plastic_seed = 1, boat_number = 2, speed_factor_boat=3, animation = False, weighted_alpha_value = 0.5 ):#, max_capacity_value=6000, resting_hours_amount=12

    # Initiate

    # Logging konfigurieren
    logging.basicConfig(level=logging.CRITICAL)#info
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
    b = GreedyBoat(loglevel=logging.INFO, patches_model=o,  weighted_alpha = weighted_alpha_value) # , max_capacity= max_capacity_value, resting_hours= resting_hours_amount
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
    print(b.history)

    if animation == True:
        try:
            animation_custom(model = o,fast = True, compare= b,size='value', show_trajectories=False)
            b.plot(fast = True, show_trajectories=True) # zeigt die routen der boote an
        except Exception as e:
            print(f"Fehler bei der Animation: {e}")

    return short_logbook_for_graph


if __name__ == "__main__":
    run_greedy(
         time_frame = 50,
         plastic_radius = 10,
         plastic_number = 1000,
         plastic_seed = 1000,
         boat_number = 2,
         speed_factor_boat = 3,
         weighted_alpha_value =0.7, # 0 Value -> 1 Distanz
         animation = True)