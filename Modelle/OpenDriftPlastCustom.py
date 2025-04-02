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
        ('shape', {'dtype': np.float32, 'units': '1', 'description': 'Shape factor of the patch'}),
        ('elasticity', {'dtype': np.float32, 'units': '1', 'description': 'Elasticity factor of the patch'}),
        ('exposure', {'dtype': np.float32, 'units': '1', 'description': 'Instantaneous flow exposure', 'default': 0.0}),
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

            exposure = props['drift_factor'] / (1.0 + 0.1 * props['patch_weight'] + 0.1 * props['patch_area'] + 0.1 * props['patch_shape'] + 0.1 * props['patch_elasticity'])

            drag_coefficient = 0.47 * props['patch_shape']  # assume spherical reference
            surface_area_ratio = props['patch_area'] / max(0.01, props['patch_weight'])

            self.seed_elements(
                lon=rand_lon, lat=rand_lat, time=time, number=1,
                current_drift_factor=props['drift_factor'],
                density=props['patch_density'],
                weight=props['patch_weight'],
                area=props['patch_area'],
                shape=props['patch_shape'],
                elasticity=props['patch_elasticity'],
                exposure=exposure,
                drag_coefficient=drag_coefficient,
                surface_area_ratio=surface_area_ratio
            )

    def update(self):
        self.advect_ocean_current()
        self.stokes_drift()
        self.merge_close_patches()

    def advect_ocean_current(self):
        u = self.environment.x_sea_water_velocity * self.elements.current_drift_factor
        v = self.environment.y_sea_water_velocity * self.elements.current_drift_factor

        drag = self.elements.drag_coefficient * self.elements.surface_area_ratio
        modifier = 1.0 / (1.0 + 0.1 * self.elements.weight + 0.1 * self.elements.area + drag)
        u *= modifier
        v *= modifier

        self.elements.exposure = np.sqrt(u**2 + v**2)

        logger.debug(f"u mean: {np.mean(u):.4f}, v mean: {np.mean(v):.4f}, modifier mean: {np.mean(modifier):.4f}")

        self.update_positions(u, v)

    def stokes_drift(self):
        if not self.get_config('drift:stokes_drift'):
            return
        u = self.environment.sea_surface_wave_stokes_drift_x_velocity
        v = self.environment.sea_surface_wave_stokes_drift_y_velocity

        drag = self.elements.drag_coefficient * self.elements.surface_area_ratio
        modifier = 1.0 / (1.0 + 0.1 * self.elements.weight + 0.1 * self.elements.area + drag)
        u *= modifier
        v *= modifier

        self.elements.exposure += np.sqrt(u**2 + v**2)

        logger.debug(f"Stokes u mean: {np.mean(u):.4f}, v mean: {np.mean(v):.4f}, modifier mean: {np.mean(modifier):.4f}")

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
                    self.elements.shape[i] = (self.elements.shape[i] + self.elements.shape[j]) / 2
                    self.elements.elasticity[i] = (self.elements.elasticity[i] + self.elements.elasticity[j]) / 2
                    self.elements.current_drift_factor[i] = (self.elements.current_drift_factor[i] + self.elements.current_drift_factor[j]) / 2
                    self.elements.density[i] = (self.elements.density[i] + self.elements.density[j]) / 2
                    self.elements.drag_coefficient[i] = (self.elements.drag_coefficient[i] + self.elements.drag_coefficient[j]) / 2
                    self.elements.surface_area_ratio[i] = (self.elements.surface_area_ratio[i] + self.elements.surface_area_ratio[j]) / 2
                    self.elements.exposure[i] = self.elements.current_drift_factor[i] / (1.0 + 0.1 * self.elements.weight[i] + 0.1 * self.elements.area[i] + 0.1 * self.elements.shape[i] + 0.1 * self.elements.elasticity[i])
                    self.elements.lat[j] = np.nan
                    self.elements.lon[j] = np.nan
                    merged.add(j)

        if merged:
            self.deactivate_elements(np.isnan(self.elements.lat))