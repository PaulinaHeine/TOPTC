from datetime import datetime, timedelta
import numpy as np
import logging; logger = logging.getLogger(__name__)
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from data.Patches.patch_composition import generate_random_patch

import random

class Lagrangian3DArray(LagrangianArray):
    variables = LagrangianArray.add_variables([
        ('current_drift_factor', {'dtype': np.float32, 'units': '1', 'description': 'Drift factor for currents', 'default': 1.}),
        ('density', {'dtype': np.float32, 'units': 'kg/m^3', 'description': 'Density'}),
        ('weight', {'dtype': np.float32, 'units': 'kg', 'description': 'Total weight of patch'}),
        ('area', {'dtype': np.float32, 'units': 'm^2', 'description': 'Total area of patch'}),
        ('drag_coefficient', {'dtype': np.float32, 'units': '1', 'description': 'Hydrodynamic drag coefficient', 'default': 1.0}),
        ('surface_area_ratio', {'dtype': np.float32, 'units': '1', 'description': 'Surface area to volume ratio', 'default': 1.0})
    ])

class OpenDriftPlastCustom(OpenDriftSimulation):
    ElementType = Lagrangian3DArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_height': {'fallback': 0},
        'sea_surface_wave_stokes_drift_x_velocity': {'fallback': 0},
        'sea_surface_wave_stokes_drift_y_velocity': {'fallback': 0},
        'sea_floor_depth_below_sea_level': {'fallback': 10000},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._add_config({
            'drift:stokes_drift': {'type': 'bool', 'default': True,
                                   'description': 'Use Stokes drift',
                                   'level': 3},
            'drift:current_drift_factor': {'type': 'float', 'default': 1.0,
                                           'min': 0.0, 'max': 5.0,
                                           'units': '1',
                                           'description': 'Factor to scale ocean current drift',
                                           'level': 3},
        })

    def seed_plastic_patch(self, lon, lat, time, number=1, radius_km=5):
        for _ in range(number):
            patch = generate_random_patch()
            props = patch['properties']
            logger.info(f"Seeding patch with properties: {props}")

            km_to_deg = 1.0 / 111.0
            rand_lat = lat + (random.uniform(-1, 1) * radius_km * km_to_deg)
            rand_lon = lon + (random.uniform(-1, 1) * radius_km * km_to_deg / np.cos(np.radians(lat)))

            density_factor = max(0.01, (1.025 - props['patch_density']) / 1.025)
            area_factor = props['patch_area'] / 10.0
            weight_factor = 1.0 / (1.0 + props['patch_weight'])

            drift_factor = np.sqrt(density_factor * area_factor * weight_factor)
            drift_factor = np.clip(drift_factor, 0.01, 0.2)

            drag_coefficient = 0.47 * (1.0 + 0.5 * props['patch_density'])
            drag_coefficient = np.clip(drag_coefficient, 0.1, 2.0)

            surface_area_ratio = props['patch_area'] / max(0.01, props['patch_weight'])

            self.seed_elements(
                lon=rand_lon, lat=rand_lat, time=time, number=1,
                current_drift_factor=drift_factor,
                density=props['patch_density'],
                weight=props['patch_weight'],
                area=props['patch_area'],
                drag_coefficient=drag_coefficient,
                surface_area_ratio=surface_area_ratio
            )

    def update(self):
        self.advect_ocean_current()
        self.stokes_drift()
        self.merge_close_patches()

    def advect_ocean_current(self):
        u_rel = self.environment.x_sea_water_velocity * self.elements.current_drift_factor
        v_rel = self.environment.y_sea_water_velocity * self.elements.current_drift_factor
        speed = np.sqrt(u_rel**2 + v_rel**2)
        max_speed = 0.05
        scale = np.clip(max_speed / (speed + 1e-8), 0, 1.0)
        u = u_rel * scale
        v = v_rel * scale
        self.update_positions(u, v)

    def stokes_drift(self):
        if not self.get_config('drift:stokes_drift'):
            return
        u_rel = self.environment.sea_surface_wave_stokes_drift_x_velocity * self.elements.current_drift_factor
        v_rel = self.environment.sea_surface_wave_stokes_drift_y_velocity * self.elements.current_drift_factor
        speed = np.sqrt(u_rel**2 + v_rel**2)
        max_speed = 0.05
        scale = np.clip(max_speed / (speed + 1e-8), 0, 1.0)
        u = u_rel * scale
        v = v_rel * scale
        self.update_positions(u, v)

    def merge_close_patches(self, threshold_km=1.0):
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