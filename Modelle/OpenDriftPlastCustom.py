from datetime import datetime, timedelta
import numpy as np
import logging;



logger = logging.getLogger(__name__)
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from data.Patches.patch_composition import generate_random_patch, generate_test_patch, generate_static_patch
import math
from datetime import datetime
import random
import math


import cartopy.crs as ccrs
import matplotlib
matplotlib.rcParams['legend.numpoints'] = 1
matplotlib.rcParams['legend.scatterpoints'] = 1
matplotlib.rcParams['figure.autolayout'] = True
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.path import Path



def offset_point(lat, lon, offset_m=100):
    """Verschiebt (lat, lon) um festen Abstand offset_m in zufälliger Richtung."""
    R = 6371000.0
    delta_deg = offset_m / 111000.0
    bearing = random.uniform(0, 2 * math.pi)
    new_lat = lat + delta_deg * math.cos(bearing)
    new_lon = lon + delta_deg * math.sin(bearing) / math.cos(math.radians(lat))
    return new_lat, new_lon


class Lagrangian3DArray(LagrangianArray):
    variables = LagrangianArray.add_variables([
        ('current_drift_factor', {'dtype': np.float32, 'units': '1', 'description': 'Drift factor for currents', 'default': 1.}),
        ('density', {'dtype': np.float32, 'units': 'kg/m^3', 'description': 'Density'}),
        ('weight', {'dtype': np.float32, 'units': 'kg', 'description': 'Total weight of patch'}),
        ('area', {'dtype': np.float32, 'units': 'm^2', 'description': 'Total area of patch'}),
        ('drag_coefficient', {'dtype': np.float32, 'units': '1', 'description': 'Hydrodynamic drag coefficient', 'default': 1.0}),
        ('surface_area_ratio', {'dtype': np.float32, 'units': '1', 'description': 'Surface area to volume ratio', 'default': 1.0}),
        ('markersize', {'dtype': np.float32, 'units': '1', 'description': 'Size for plotting', 'default': 20.0}),
        ('value', {'dtype': np.float32, 'units': '1', 'description': 'Reward value of patch', 'default': 1.0}),
        ('patch_id', {'dtype': np.int32, 'units': '1', 'description': 'Unique Patch ID', 'default': -1}),
        ('is_patch', {'dtype': np.bool_, 'units': '1', 'description': 'True if element is a patch', 'default': True}),


    ])

class OpenDriftPlastCustom(OpenDriftSimulation):
    ElementType = Lagrangian3DArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0}
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self._add_config({
            'drift:stokes_drift': {'type': 'bool', 'default': False,
                                   'description': 'Use Stokes drift',
                                   'level': 3},
            'drift:current_drift_factor': {'type': 'float', 'default': 1.0,
                                           'min': 0.0, 'max': 5.0,
                                           'units': '1',
                                           'description': 'Factor to scale ocean current drift',
                                           'level': 3},
        })
        self.next_patch_id = 1 # pacht id startet bei 1, nicht bei 0
        self.custom_history = []

    def seed_plastic_patch(self, lon, lat, time, number=1, radius_km=5, z=1, seed=1):
        for i in range(number):
            combined_seed = seed + i

            # 1. Patch-Eigenschaften eindeutig und deterministisch
            patch = generate_static_patch(seed=combined_seed)
            props = patch['properties']

            logger.info(f"Seeding patch with properties: {props}")

            # 2. Position eindeutig und deterministisch
            random.seed(combined_seed)
            km_to_deg = 1.0 / 111.0
            rand_lat = lat + (random.uniform(-1, 1) * radius_km * km_to_deg)
            rand_lon = lon + (random.uniform(-1, 1) * radius_km * km_to_deg / np.cos(np.radians(lat)))

            # 3. Physikalische Ableitungen (keine Zufallswerte!)
            density_factor = max(0.01, (1.025 - props['patch_density']) / 1.025)
            area_factor = props['patch_area'] / 10.0
            weight_factor = 1.0 / (1.0 + props['patch_weight'])

            drift_factor = np.clip(np.sqrt(density_factor * area_factor * weight_factor), 0.01, 0.2)
            drag_coefficient = np.clip(0.47 * (1.0 + 0.5 * props['patch_density']), 0.1, 2.0)
            surface_area_ratio = props['patch_area'] / max(0.01, props['patch_weight'])
            markersize = np.clip(props['patch_area'] * 100, 10, 300)
            value = (props['patch_area'] * props['patch_density'] * props['patch_weight']) / 100

            self.seed_elements(
                lon=rand_lon, lat=rand_lat, time=time, number=1,
                current_drift_factor=drift_factor,
                density=props['patch_density'],
                weight=props['patch_weight'],
                area=props['patch_area'],
                z=z,
                value=value,
                drag_coefficient=drag_coefficient,
                surface_area_ratio=surface_area_ratio,
                markersize=markersize,
                patch_id=self.next_patch_id
            )
            self.next_patch_id += 1
            self.release_elements()

    def update(self):

        super().update()

        self.environment = self.get_environment(
                                variables=['x_sea_water_velocity', 'y_sea_water_velocity'],
                                time=self.time,           # aktueller Zeitpunkt
                                lon=self.elements.lon,    # aktuelle Längengrade
                                lat=self.elements.lat,      # aktuelle Breitengrade
                                z = self.elements.z,
                                profiles=None
                                )[0]
        self.advect_ocean_current()
        #self.merge_close_patches()
        self.record_custom_history()
        self.elements.age_seconds += self.time_step.total_seconds()


    def advect_ocean_current(self):
        u_rel = self.environment.x_sea_water_velocity * self.elements.current_drift_factor
        v_rel = self.environment.y_sea_water_velocity * self.elements.current_drift_factor
        speed = np.sqrt(u_rel**2 + v_rel**2)
        max_speed = 0.05
        scale = np.clip(max_speed / (speed + 1e-8), 0, 1.0)
        u = u_rel * scale
        v = v_rel * scale
        if len(u_rel) == 0:
            logger.warning("⚠️ Keine Strömungsdaten vorhanden – Bewegung wird übersprungen.")
            return
        self.update_positions(u, v)


    def merge_close_patches(self, threshold_km=0.1):# TODO soll abhängig von der größe sein
        threshold_deg = threshold_km / 111.0
        positions = np.vstack([self.elements.lat, self.elements.lon]).T
        merged = set()

        for i in range(len(positions)):
            if i in merged:
                continue
            for j in range(i + 1, len(positions)):
                if j in merged:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < threshold_deg:
                    self.elements.weight[i] += self.elements.weight[j]
                    self.elements.area[i] += self.elements.area[j]
                    self.elements.current_drift_factor[i] = (self.elements.current_drift_factor[i] + self.elements.current_drift_factor[j]) / 2
                    self.elements.density[i] = (self.elements.density[i] + self.elements.density[j]) / 2
                    self.elements.drag_coefficient[i] = (self.elements.drag_coefficient[i] + self.elements.drag_coefficient[j]) / 2
                    self.elements.surface_area_ratio[i] = (self.elements.surface_area_ratio[i] + self.elements.surface_area_ratio[j]) / 2
                    self.elements.lat[j] = np.nan
                    self.elements.lon[j] = np.nan
                    merged.add(j)

        if merged:
            self.deactivate_elements(np.isnan(self.elements.lat))

    def remove_large_patches_randomly(self):
        """
        Löscht zufällig große/schwere Patches ohne Ersatz.
        Kein Split, kein Seeding, nur Entfernung.
        """
        if self.num_elements_active() == 0:
            return

        base_removal_probability = 0.001
        area_threshold = 100.0
        weight_threshold = 300.0

        removed = 0

        for i in range(self.num_elements_active()):
            weight = self.elements.weight[i]
            area = self.elements.area[i]

            if area < area_threshold and weight < weight_threshold:
                continue

            if np.random.rand() > base_removal_probability:
                continue

            # Markiere Patch zum Entfernen
            self.elements.lat[i] = np.nan
            self.elements.lon[i] = np.nan
            removed += 1

            logger.info(f"Patch {i} entfernt (area={area:.1f}, weight={weight:.1f})")

        if removed > 0:
            self.deactivate_elements(np.isnan(self.elements.lat))
            logger.info(f"{removed} große Patches wurden zufällig entfernt.")


    def record_custom_history(self):
        if not hasattr(self, 'custom_history_list'):
            self.custom_history_list = []

        num_elements = self.num_elements_total()

        for i in range(num_elements):
            patch_id = int(self.elements.patch_id[i])
            entry = (
                int(self.elements.patch_id[i]), self.elements.status[i], self.elements.moving[i], float(self.elements.age_seconds[i]), i,
                round(float(self.elements.lon[i]), 8),
                round(float(self.elements.lat[i]), 8),
                float(self.elements.z[i]),
                float(self.elements.density[i]),
                float(self.elements.drag_coefficient[i]),
                float(self.elements.area[i]),
                float(self.elements.weight[i]),
                float(self.elements.current_drift_factor[i]),
                float(self.elements.surface_area_ratio[i]),
                float(self.elements.value[i]),
                float(self.environment.x_sea_water_velocity[i]),
                float(self.environment.y_sea_water_velocity[i])
            )


            # sicherstellen, dass Index für patch_id existiert
            while len(self.custom_history_list) <= patch_id - 1:
                self.custom_history_list.append([])

            if entry[5] == "nan":
                self.custom_history_list[-1].append(entry)
                break
            else:
                self.custom_history_list[patch_id - 1].append(entry)

    def get_structured_history(self):
        import numpy as np
        import numpy.ma as ma

        dtype = [
            ('ID', 'i4'), ('unused1', 'i4'), ('unused2', 'i4'), ('age', 'f8'), ('idx', 'i4'),
            ('lon', 'f8'), ('lat', 'f8'), ('z', 'f4'), ('density', 'f4'),
            ('drag', 'f4'), ('area', 'f4'), ('weight', 'f4'), ('drift_factor', 'f4'),
            ('surface_ratio', 'f4'), ('value', 'f4'),
            ('u', 'f4'), ('v', 'f4')
        ]

        all_records = []
        for patch in self.custom_history_list:
            arr = np.array(patch, dtype=dtype)
            all_records.append(ma.masked_array(arr))

        return ma.stack(all_records)


