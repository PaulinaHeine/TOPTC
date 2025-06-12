import numpy as np
import logging
from opendrift.models.basemodel import OpenDriftSimulation
from opendrift.elements import LagrangianArray
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class GreedyBoatArray(LagrangianArray):
    variables = LagrangianArray.add_variables([
        ('speed_factor', {'dtype': np.float32, 'units': '1', 'description': 'Base speed factor', 'default': 1.0}),
        ('target_lon', {'dtype': np.float32, 'units': 'deg', 'description': 'Target longitude'}),
        ('target_lat', {'dtype': np.float32, 'units': 'deg', 'description': 'Target latitude'}),
        ('current_drift_factor', {'dtype': np.float32, 'units': '1', 'description': 'For compatibility', 'default': 10.0}),
        ('is_patch', {'dtype': np.bool_, 'units': '1', 'description': 'True if element is a patch', 'default': False}),
        ('target_patch_index', {'dtype': np.int32, 'units': '1', 'description': 'Index of target patch', 'default': -1}),
        ('collected_value', {'dtype': np.float32, 'units': '1', 'description': 'Total value collected by the boat', 'default': 0.0}),
        #('capacity', {'dtype': np.float32, 'units': 'kg', 'default': 5000.0}),
        #('in_rest', {'dtype': np.bool_, 'default': False}),
        #('resting_hours_left',{'dtype': np.float32, 'units': 'h', 'description': 'Remaining resting hours', 'default': 0.0}),
        ('distance_traveled', {'dtype': np.float32, 'units': 'km', 'default': 0.0}),

    ])


class GreedyBoat(OpenDriftSimulation):
    ElementType = GreedyBoatArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
    }

    def __init__(self, patches_model, target_mode, weighted_alpha = None, *args, **kwargs): # capacity, resting hours
        super().__init__(*args, **kwargs)
        self.patches_model = patches_model
        self.target_mode = target_mode
        if self.target_mode == "weighted":
            self.weighted_alpha = weighted_alpha
        #self.capacity = capacity
        #self.resting_hours = resting_hours

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
        self.check_and_pick_new_target()
        self.move_toward_target()
        self.record_custom_history()
        #self.validate_boats()

    def validate_boats(self): # only when errer occurs
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
                logger.warning(f"❌ Boot {i} hat ungültige Position: ({lat}, {lon})")
            if tgt == -1:
                no_target += 1
            if speed == 0.0 and tgt == -1:
                stuck += 1

        logger.info(
            f"📊 Boot-Status: total={total}, active={active}, NaN-pos={nan_pos}, no-target={no_target}, stuck={stuck}")

    def move_toward_target(self):
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

            if dist < (0.1 / 111.0):
                self.deactivate_patch_near(self.elements.lat[i], self.elements.lon[i], boat_idx=i)
                self.elements.target_patch_index[i] = -1

                if self.target_mode == "value":
                    self.assign_target_value(i)
                elif self.target_mode == "weighted":
                    self.assign_target_weighted(i)
                elif self.target_mode == "distance":
                    self.assign_target_distance(i)
                continue

            dlon_norm = dlon / dist
            dlat_norm = dlat / dist

            # Schrittgröße
            max_step_deg = (0.06 / 111.0) * self.elements.speed_factor[i]
            step_deg = min(dist, max_step_deg)
            step_lon = dlon_norm * step_deg
            step_lat = dlat_norm * step_deg

            # 🌊 Strömungseinfluss
            try:
                u = self.environment['x_sea_water_velocity'][i]
                v = self.environment['y_sea_water_velocity'][i]
            except Exception as e:
                logger.warning(f"⚠️ Strömungsdaten nicht verfügbar für Boot {i}: {e}")
                u, v = 0.0, 0.0

            drift_scale = 3600 / 111000  # m/s → °/h (1h Zeitschritt)
            drift_lon = u * drift_scale
            drift_lat = v * drift_scale

            drift_influence = 0.1 #####!!!!! WICHTIG EINFLUSSFAKTOR DER STRÖMUNG AUF BOOT
            step_lon += drift_influence * drift_lon
            step_lat += drift_influence * drift_lat

            self.elements.lon[i] += step_lon
            self.elements.lat[i] += step_lat

            self.elements.distance_traveled[i] += np.sqrt(step_lon ** 2 + step_lat ** 2) * 111.0

    def seed_boat(self, lon, lat, number=1, time=None, speed_factor=1.0):
        # 1. Anzahl Elemente vor dem Seeden merken
        pre_count = self.num_elements_total()

        # 2. Boote seeden (in den Buffer)
        self.seed_elements(
            lon=lon,
            lat=lat,
            number=number,
            time=time,
            target_lon=lon,
            target_lat=lat,
            speed_factor=speed_factor
        )

        # 3. Anzahl nach dem Seeden
        post_count = self.num_elements_total()
        newly_seeded = slice(pre_count, post_count)

        # 4. Speed-Faktor korrekt auf neue Boote anwenden
        self.elements.speed_factor[newly_seeded] = speed_factor

        # 5. Aktivieren
        self.release_elements()

        logger.info(f"⚓ {number} Boot(e) geseedet bei ({lat}, {lon}) mit speed_factor={speed_factor}")

    def check_and_pick_new_target(self, threshold_km=0.1):
        #threshold_deg = threshold_km / 111.0
        for i in range(self.num_elements_active()):
            if self.elements.target_patch_index[i] == -1:
                if self.target_mode == "value":
                    self.assign_target_value(i)
                elif self.target_mode == "weighted":
                    self.assign_target_weighted(i)
                elif self.target_mode == "distance":
                    self.assign_target_distance(i)


    def deactivate_patch_near(self, lat, lon, boat_idx=None, radius_km=0.100):
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
                if boat_idx is not None:
                    self.elements.collected_value[boat_idx] += self.patches_model.elements.value[i]

                self.patches_model.elements.value[i] = 0.0
                self.patches_model.elements.lat[i] = np.nan
                self.patches_model.elements.lon[i] = np.nan
                logger.info(f"🧹 Patch {i} deaktiviert")

        self.patches_model.deactivate_elements(np.isnan(self.patches_model.elements.lat))

##################################################################################################
############                      OPTIMIZATION FUNCTIONS                    ######################
##################################################################################################

    def assign_target_value(self, boat_idx):
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

    def assign_target_distance(self, boat_idx):
        if self.patches_model.num_elements_active() == 0:
            logger.info(f"⚓ Boot {boat_idx}: Keine Ziele mehr verfügbar.")
            return

        if self.elements.target_patch_index[boat_idx] != -1:
            return  # Boot hat bereits ein Ziel

        min_dist = float('inf')
        best_idx = -1

        boat_lon = self.elements.lon[boat_idx]
        boat_lat = self.elements.lat[boat_idx]
        taken_targets = set(self.elements.target_patch_index[:self.num_elements_active()])

        for i in range(self.patches_model.num_elements_total()):
            if self.patches_model.elements.status[i] != 0:
                continue
            if not self.patches_model.elements.is_patch[i]:
                continue
            if i in taken_targets:
                continue

            dlon = self.patches_model.elements.lon[i] - boat_lon
            dlat = self.patches_model.elements.lat[i] - boat_lat
            dist = np.sqrt(dlon ** 2 + dlat ** 2)

            if dist < min_dist:
                min_dist = dist
                best_idx = i

        if best_idx == -1:
            logger.info(f"🛑 Boot {boat_idx}: Kein unbesetzter Patch verfügbar.")
            return

        self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[best_idx]
        self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[best_idx]
        self.elements.target_patch_index[boat_idx] = best_idx

        logger.info(f"🎯 Boot {boat_idx} visiert Patch {best_idx} an (Distanz = {min_dist:.4f}°)")
        # Todo wenn distanz zu groß wird: neuer patch wird anvisiert

    def assign_target_weighted(self, boat_idx):
        """
        α (alpha) nahe 1 ⇒ Distanz ist wichtiger
        → Du bevorzugst nahegelegene Patches, selbst wenn ihr Value niedrig ist.
        α nahe 0 ⇒ Value ist wichtiger
        → Du bevorzugst Patches mit hohem Wert, auch wenn sie weiter weg sind.
        """
        if self.patches_model.num_elements_active() == 0:
            logger.info(f"⚓ Boot {boat_idx}: Keine Ziele mehr verfügbar.")
            return

        if self.elements.target_patch_index[boat_idx] != -1:
            return  # Boot hat bereits ein Ziel

        boat_lon = self.elements.lon[boat_idx]
        boat_lat = self.elements.lat[boat_idx]
        taken_targets = set(self.elements.target_patch_index[:self.num_elements_active()])

        candidates = []

        for i in range(self.patches_model.num_elements_total()):
            if self.patches_model.elements.status[i] != 0:
                continue
            if not self.patches_model.elements.is_patch[i]:
                continue
            if i in taken_targets:
                continue

            patch_value = self.patches_model.elements.value[i]
            dlon = self.patches_model.elements.lon[i] - boat_lon
            dlat = self.patches_model.elements.lat[i] - boat_lat
            dist = np.sqrt(dlon ** 2 + dlat ** 2)

            candidates.append((i, dist, patch_value))

        if not candidates:
            logger.info(f"🛑 Boot {boat_idx}: Keine geeigneten Patches.")
            return

        # Normierung vorbereiten
        max_dist = max([c[1] for c in candidates])
        max_value = max([c[2] for c in candidates]) or 1.0  # avoid division by zero

        # Scoring
        scored = []
        for i, dist, value in candidates:
            norm_dist = 1 - dist / max_dist  # näher = besser
            norm_value = value / max_value  # höher = besser
            score = self.weighted_alpha * norm_dist + (1 - self.weighted_alpha) * norm_value
            scored.append((i, score))

        best_idx, best_score = max(scored, key=lambda x: x[1])

        self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[best_idx]
        self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[best_idx]
        self.elements.target_patch_index[boat_idx] = best_idx

        logger.info(f"🎯 Boot {boat_idx} visiert Patch {best_idx} an (Score = {best_score:.4f}, α = {self.weighted_alpha})")

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
                float(self.elements.target_lat[i]),
                float(self.elements.collected_value[i]),
                float(self.elements.distance_traveled[i]),
                int(self.elements.target_patch_index[i]),
            )

            while len(self.custom_history_list) <= i:
                self.custom_history_list.append([])

            self.custom_history_list[i].append(entry)

    def get_structured_history(self):
        import numpy.ma as ma

        dtype = [
            ('ID', 'i4'), ('status', 'i4'), ('moving', 'i4'), ('age', 'f8'), ('idx', 'i4'),
            ('lon', 'f8'), ('lat', 'f8'), ('speed_factor', 'f4'),
            ('target_lon', 'f8'), ('target_lat', 'f8'), ('collected_value', 'f4'),('distance_traveled', 'f4'),  ('target_patch_index', 'i4')
        ]

        all_records = []
        for boat in self.custom_history_list:
            arr = np.array(boat, dtype=dtype)
            all_records.append(ma.masked_array(arr))

        return ma.stack(all_records)

    def print_collection_summary(self):
        print("\n🚤 Boot-Sammelübersicht:")
        total_value = 0
        total_distance = 0
        total_collected = 0

        for i in range(self.num_elements_active()):
            value = float(self.elements.collected_value[i])
            distance = float(self.elements.distance_traveled[i])
            if value > 0:
                total_collected += 1
            total_value += value
            total_distance += distance

            print(f"\n🚤 Boot {i}:")
            print(f"   📦 Gesammelter Wert: {value:.2f}")
            print(f"   📍 Gefahrene Strecke: {distance:.2f} km")
            print(f"   🧩 Gesammelte Patches:")

        for boat_idx, history in enumerate(self.custom_history_list):
            print(f"\n🚤 Boot {boat_idx}:")

            hour = 0
            prev = None



            for entry in history:
                (ID, status, moving, age, idx, lon, lat, speed, t_lon, t_lat,
                 value, dist, patch_idx) = entry

                # Nur wenn sich was Wesentliches ändert
                if prev is None or (
                        value != prev[10] or
                        patch_idx != prev[12] or
                        #abs(dist - prev[11]) > 0.01 or
                        status != prev[1]
                ):
                    print(
                        f"  ⏱ Stunde {hour}: "
                        f"🏁 Gesammelt: {value:.2f}, "
                        f"🛣️ Strecke: {dist:.2f} km"
                    )
                prev = entry
                hour += 1

        print(f"\n📦 Gesamtwert aller Boote: {total_value:.2f}")
        print(f"🚗 Gesamtstrecke aller Boote: {total_distance:.2f} km")
        print(f"✅ Aktive Boote mit Sammlung: {total_collected} von {self.num_elements_active()}")



        # flexibel assignene, wenn wert für anderes boot höhr wäre, switchen