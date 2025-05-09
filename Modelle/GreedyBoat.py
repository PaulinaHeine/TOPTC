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
        ('is_patch', {'dtype': np.bool_, 'units': '1', 'description': 'True if element is a patch', 'default': False}),
        ('target_patch_index', {'dtype': np.int32, 'units': '1', 'description': 'Index of target patch', 'default': -1}),

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
        super().update()
        self.move_toward_target()
        self.check_and_pick_new_target()
        self.record_custom_history()

    def seed_boat(self, lon, lat, number=1, time=None, speed_factor=1.0):
        """Platziere ein Boot zu Beginn der Simulation."""
        self.seed_elements(
            lon=lon,
            lat=lat,
            number=number,
            time=time,
            target_lon=lon,  # direkt mitgeben
            target_lat=lat,  # direkt mitgeben
        )
        self.elements.speed_factor[:] = speed_factor
        logger.info(f"⚓ Boot geseedet bei ({lat}, {lon})")
        self.release_elements()

    def move_toward_target(self):

        patch_idx = self.elements.target_patch_index

        self.elements.target_lon = self.patches_model.elements.lon[patch_idx]
        self.elements.target_lat = self.patches_model.elements.lat[patch_idx]

        dlon = self.elements.target_lon - self.elements.lon
        dlat = self.elements.target_lat - self.elements.lat
        dist = np.sqrt(dlon**2 + dlat**2)

        dlon_norm = dlon / (dist + 1e-8)
        dlat_norm = dlat / (dist + 1e-8)

        step_deg = (0.06 / 111.0) * self.elements.speed_factor

        self.elements.lon += dlon_norm * step_deg
        self.elements.lat += dlat_norm * step_deg



    def check_and_pick_new_target(self, threshold_km=0.1):
        threshold_deg = threshold_km / 111.0

        for i in range(self.num_elements_active()):
            d = np.sqrt(
                (self.elements.lon[i] - self.elements.target_lon[i]) ** 2 +
                (self.elements.lat[i] - self.elements.target_lat[i]) ** 2
            )
            if d < threshold_deg:
                logger.info(f" Boot {i} hat Ziel erreicht")


                # In check_and_pick_new_target:
                self.deactivate_patch_near(self.elements.lat[i], self.elements.lon[i])

                self.assign_target(i)

    def deactivate_patch_near(self, lat, lon, radius_km=0.1):
        threshold_deg = radius_km / 111.0
        num = self.patches_model.num_elements_total()

        to_deactivate = np.zeros(num, dtype=bool)

        for i in range(num):
            if self.patches_model.elements.status[i] != 0:
                continue  # nur aktive

            if not self.patches_model.elements.is_patch[i]:
                continue  # nur Patches, keine Boote

            d = np.sqrt(
                (self.patches_model.elements.lat[i] - lat) ** 2 +
                (self.patches_model.elements.lon[i] - lon) ** 2
            )

            if d < threshold_deg:
                # Patch als "deaktiviert" markieren
                self.patches_model.elements.value[i] = 0.0
                self.patches_model.elements.lat[i] = np.nan
                self.patches_model.elements.lon[i] = np.nan
                logger.info(f"🧹 Patch {i} deaktiviert")


        # Deaktivieren aller NaN-Patches (wie in deinem Beispiel)
        self.patches_model.deactivate_elements(np.isnan(self.patches_model.elements.lat))


    def assign_target(self, boat_idx):
        if self.patches_model.num_elements_active() == 0:
            logger.info(f"⚓ Boot {boat_idx}: Keine Ziele mehr verfügbar.")
            return

        values = self.patches_model.elements.value[:self.patches_model.num_elements_active()]
        i_max = np.argmax(values)


        self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[i_max]
        self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[i_max]
        self.elements.target_patch_index[boat_idx] = i_max


        logger.info(f"Boot {boat_idx} visiert Patch {i_max} an (value = {values[i_max]:.2f})")

    def record_custom_history(self):
        if not hasattr(self, 'custom_history_list'):
            self.custom_history_list = []

        num_elements = self.num_elements_total()
        for i in range(num_elements):
            entry = (
                int(i),  # ID
                self.elements.status[i],
                self.elements.moving[i],
                float(self.elements.age_seconds[i]),
                i,
                float(self.elements.lon[i]),
                float(self.elements.lat[i]),
                float(self.elements.speed_factor[i]),
                float(self.elements.target_lon[i]),
                float(self.elements.target_lat[i])
            )

            while len(self.custom_history_list) <= i:
                self.custom_history_list.append([])

            self.custom_history_list[i].append(entry)

    def get_structured_history(self):
        import numpy as np
        import numpy.ma as ma

        dtype = [
            ('ID', 'i4'), ('status', 'i4'), ('moving', 'i4'), ('age', 'f8'), ('idx', 'i4'),
            ('lon', 'f8'), ('lat', 'f8'), ('speed_factor', 'f4'),
            ('target_lon', 'f8'), ('target_lat', 'f8')
        ]

        all_records = []
        for boat in self.custom_history_list:
            arr = np.array(boat, dtype=dtype)
            all_records.append(ma.masked_array(arr))

        return ma.stack(all_records)

#Todo manchmal sieht der weg des boots weird aus warum amcht der solche schlenker