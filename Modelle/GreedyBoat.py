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

    def move_toward_target(self):
        for i in range(self.num_elements_active()):
            patch_idx = int(self.elements.target_patch_index[i])
            if patch_idx < 0 or patch_idx >= self.patches_model.num_elements_total():
                continue

            target_lon = self.patches_model.elements.lon[patch_idx]
            target_lat = self.patches_model.elements.lat[patch_idx]

            dlon = target_lon - self.elements.lon[i]
            dlat = target_lat - self.elements.lat[i]
            dist = np.sqrt(dlon**2 + dlat**2)

            if dist < (0.1 / 111.0):
                self.deactivate_patch_near(self.elements.lat[i], self.elements.lon[i])
                self.elements.target_patch_index[i] = -1
                self.assign_target(i)
                continue

            dlon_norm = dlon / dist
            dlat_norm = dlat / dist

            step_deg = (0.06 / 111.0) * self.elements.speed_factor[i]

            self.elements.lon[i] += dlon_norm * step_deg
            self.elements.lat[i] += dlat_norm * step_deg

    def update(self):
        super().update()
        self.move_toward_target()
        self.check_and_pick_new_target()
        self.record_custom_history()

    def seed_boat(self, lon, lat, number=1, time=None, speed_factor=1.0):
        pre_count = self.num_elements_total()
        self.seed_elements(
            lon=lon,
            lat=lat,
            number=number,
            time=time,
            target_lon=lon,
            target_lat=lat,
        )
        post_count = self.num_elements_total()
        newly_seeded = slice(pre_count, post_count)
        self.elements.speed_factor[newly_seeded] = speed_factor
        logger.info(f"⚓ {number} Boot(e) geseedet bei ({lat}, {lon})")
        self.release_elements()

    def check_and_pick_new_target(self, threshold_km=0.1):
        threshold_deg = threshold_km / 111.0
        for i in range(self.num_elements_active()):
            if self.elements.target_patch_index[i] == -1:
                self.assign_target(i)

    def deactivate_patch_near(self, lat, lon, radius_km=0.1):
        threshold_deg = radius_km / 111.0
        num = self.patches_model.num_elements_total()

        for i in range(num):
            if self.patches_model.elements.status[i] != 0:
                continue
            if not self.patches_model.elements.is_patch[i]:
                continue

            d = np.sqrt(
                (self.patches_model.elements.lat[i] - lat) ** 2 +
                (self.patches_model.elements.lon[i] - lon) ** 2
            )

            if d < threshold_deg:
                self.patches_model.elements.value[i] = 0.0
                self.patches_model.elements.lat[i] = np.nan
                self.patches_model.elements.lon[i] = np.nan
                logger.info(f"🧹 Patch {i} deaktiviert")

        self.patches_model.deactivate_elements(np.isnan(self.patches_model.elements.lat))

    def assign_target(self, boat_idx):
        if self.patches_model.num_elements_active() == 0:
            logger.info(f"⚓ Boot {boat_idx}: Keine Ziele mehr verfügbar.")
            return

        if self.elements.target_patch_index[boat_idx] != -1:
            return  # Boot hat bereits ein Ziel

        values = self.patches_model.elements.value[:self.patches_model.num_elements_active()]
        taken_targets = set(self.elements.target_patch_index[:self.num_elements_active()])
        values = np.where([i not in taken_targets for i in range(len(values))], values, -1)

        i_max = np.argmax(values)
        if values[i_max] == -1:
            logger.info(f"🛑 Boot {boat_idx}: Kein unbesetzter Patch verfügbar.")
            return

        self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[i_max]
        self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[i_max]
        self.elements.target_patch_index[boat_idx] = i_max

        logger.info(f"🎯 Boot {boat_idx} visiert Patch {i_max} an (value = {values[i_max]:.2f})")

    def record_custom_history(self):
        if not hasattr(self, 'custom_history_list'):
            self.custom_history_list = []

        num_elements = self.num_elements_total()
        for i in range(num_elements):
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
                float(self.elements.target_lat[i])
            )

            while len(self.custom_history_list) <= i:
                self.custom_history_list.append([])

            self.custom_history_list[i].append(entry)

    def get_structured_history(self):
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
