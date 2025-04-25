import numpy as np
import logging
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray

logger = logging.getLogger(__name__)


class GreedyBoatArray(LagrangianArray):
    variables = LagrangianArray.add_variables([
        ('speed_factor', {'dtype': np.float32, 'units': '1', 'description': 'Base speed factor', 'default': 1.0}),
        ('target_lon', {'dtype': np.float32, 'units': 'deg', 'description': 'Target longitude'}),
        ('target_lat', {'dtype': np.float32, 'units': 'deg', 'description': 'Target latitude'}),
        ('current_drift_factor', {'dtype': np.float32, 'units': '1', 'description': 'For compatibility', 'default': 10.0}),
    ])



class GreedyBoat(OpenDriftSimulation):
    ElementType = GreedyBoatArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
    }

    def __init__(self, patches_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.patches_model = patches_model

    def update(self):

        self.move_toward_target()
        self.check_and_pick_new_target()

    def move_toward_target(self):
        dlon = self.elements.target_lon - self.elements.lon
        dlat = self.elements.target_lat - self.elements.lat
        dist = np.sqrt(dlon**2 + dlat**2)

        dlon_norm = dlon / (dist + 1e-8)
        dlat_norm = dlat / (dist + 1e-8)

        # Stromgeschwindigkeit holen
        u = self.environment.x_sea_water_velocity
        v = self.environment.y_sea_water_velocity

        current_mag = np.sqrt(u**2 + v**2)
        u_norm = u / (current_mag + 1e-8)
        v_norm = v / (current_mag + 1e-8)

        # Einfluss der Strömung auf Vorwärtsbewegung
        cos_theta = u_norm * dlon_norm + v_norm * dlat_norm

        # Basisbewegung + Strömungseinfluss
        step_deg = (0.06 / 111.0) * self.elements.speed_factor * (1 + 0.5 * cos_theta)

        self.elements.lon += dlon_norm * step_deg
        self.elements.lat += dlat_norm * step_deg

    def check_and_pick_new_target(self, threshold_km=0.1):
        threshold_deg = threshold_km / 111.0

        for i in range(self.num_elements_active()):
            d = np.sqrt(
                (self.elements.lon[i] - self.elements.target_lon[i])**2 +
                (self.elements.lat[i] - self.elements.target_lat[i])**2
            )
            if d < threshold_deg:
                logger.info(f"🚤 Boot {i} hat Ziel erreicht")
                self.deactivate_patch_near(self.elements.target_lat[i], self.elements.target_lon[i])
                self.assign_target(i)

    def deactivate_patch_near(self, lat, lon, radius_km=0.1):
        threshold_deg = radius_km / 111.0
        for i in range(self.patches_model.num_elements_active()):
            d = np.sqrt(
                (self.patches_model.elements.lat[i] - lat)**2 +
                (self.patches_model.elements.lon[i] - lon)**2
            )
            if d < threshold_deg:
                self.patches_model.elements.lat[i] = np.nan
                self.patches_model.elements.lon[i] = np.nan
                logger.info(f"🧹 Patch {i} deaktiviert")
        self.patches_model.deactivate_elements(np.isnan(self.patches_model.elements.lat))

    def assign_target(self, boat_idx):
        if self.patches_model.num_elements_active() == 0:
            logger.info(f"⚓ Boot {boat_idx}: Keine Ziele mehr verfügbar.")
            return

        values = self.patches_model.elements.value[:self.patches_model.num_elements_active()]
        i_max = np.argmax(values)

        self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[i_max]
        self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[i_max]

        logger.info(f"🎯 Boot {boat_idx} visiert Patch {i_max} an (value = {values[i_max]:.2f})")
