from datetime import datetime, timedelta
import numpy as np
import logging; logger = logging.getLogger(__name__)
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray

class LagrangianBoatArray(LagrangianArray):
    variables = LagrangianArray.add_variables([
        ('speed', {'dtype': np.float32, 'units': 'm/s', 'description': 'Grundgeschwindigkeit des Bootes', 'default': 1.0}),
        ('direction_x', {'dtype': np.float32, 'units': '1', 'description': 'x-Komponente der Bewegungsrichtung', 'default': 1.0}),
        ('direction_y', {'dtype': np.float32, 'units': '1', 'description': 'y-Komponente der Bewegungsrichtung', 'default': 0.0}),
        ('target_lon', {'dtype': np.float32, 'units': 'degree_east', 'description': 'Ziel-Längengrad', 'default': np.nan}),
        ('target_lat', {'dtype': np.float32, 'units': 'degree_north', 'description': 'Ziel-Breitengrad', 'default': np.nan})
    ])

class OpenDriftBoat(OpenDriftSimulation):
    ElementType = LagrangianBoatArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_target(self, target_lon, target_lat):
        self.elements.target_lon[:] = target_lon
        self.elements.target_lat[:] = target_lat

    def update(self):
        u_env = self.environment.x_sea_water_velocity
        v_env = self.environment.y_sea_water_velocity

        boat_pos = np.vstack((self.elements.lon, self.elements.lat)).T
        target_pos = np.vstack((self.elements.target_lon, self.elements.target_lat)).T

        direction_vector = target_pos - boat_pos
        distance = np.linalg.norm(direction_vector, axis=1) + 1e-6
        direction_unit = (direction_vector.T / distance).T

        # Geschwindigkeit anpassen: schneller mit Strömung, langsamer dagegen
        flow_angle = np.arctan2(v_env, u_env)
        boat_angle = np.arctan2(direction_unit[:, 1], direction_unit[:, 0])
        angle_diff = boat_angle - flow_angle
        current_speed = self.elements.speed + 0.5 * np.cos(angle_diff) * np.sqrt(u_env**2 + v_env**2)

        u_boat = direction_unit[:, 0] * current_speed
        v_boat = direction_unit[:, 1] * current_speed

        self.update_positions(u_boat, v_boat)

        # Wenn Ziel erreicht wurde, Ziel löschen
        reached = distance < 0.01  # ~1km
        self.elements.target_lon[reached] = np.nan
        self.elements.target_lat[reached] = np.nan

    def seed_boat(self, lon, lat, time, number=1):
        self.seed_elements(
            lon=lon, lat=lat, time=time, number=number,
            speed=1.0, direction_x=1.0, direction_y=0.0,
            target_lon=np.nan, target_lat=np.nan
        )
