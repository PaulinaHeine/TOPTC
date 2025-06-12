def check_capacity(self):
    for i in range(self.num_elements_active()):

        if self.elements.collected_value_current[i] >= self.max_capacity:
            self.elements.resting_hours_left[i] = self.resting_hours
            self.elements.speed_factor[i] = 0.0
            logger.info(
                f"⛔ Boot {i} erreicht Kapazität ({self.elements.collected_value_current[i]:.2f}) – ruht für {self.resting_hours}h")

        if self.elements.resting_hours_left[i] > 0:
            self.elements.resting_hours_left[i] -= 1
            if self.elements.resting_hours_left[i] <= 0:
                self.elements.collected_value_current[i] = 0.0
                self.elements.speed_factor[i] = 4.0
                logger.info(f"✅ Boot {i} ist wieder aktiv und geleert")
            # continue


def can_pickup_patch(self, boat_idx, patch_value):
    return self.elements.collected_value[boat_idx] + patch_value <= self.max_capacity


def is_resting(self, boat_idx):
    return self.elements.resting_hours_left[boat_idx] > 0



    def assign_target_value(self, boat_idx, max_capacity=None):
        if self.patches_model.num_elements_active() == 0:
            logger.info(f"⚓ Boot {boat_idx}: Keine Ziele mehr verfügbar.")
            return

        if self.elements.target_patch_index[boat_idx] != -1:
            return

        collected = self.elements.collected_value[boat_idx]
        n = self.patches_model.num_elements_active()
        values = self.patches_model.elements.value[:n]
        statuses = self.patches_model.elements.status[:n]
        is_patch = self.patches_model.elements.is_patch[:n]
        taken_targets = set(self.elements.target_patch_index[:self.num_elements_active()])

        all_valid = [(i, values[i]) for i in range(n)
                     if i not in taken_targets and statuses[i] == 0 and is_patch[i]]

        if not all_valid:
            logger.info(f"🛑 Boot {boat_idx}: Keine gültigen Patches vorhanden.")
            return

        if collected >= 0.8 * max_capacity:
            valid_within_margin = [
                (i, val) for i, val in all_valid
                if val + collected <= max_capacity
            ]
            if valid_within_margin:
                best_idx, best_val = max(valid_within_margin, key=lambda x: x[1])
                self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[best_idx]
                self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[best_idx]
                self.elements.target_patch_index[boat_idx] = best_idx
                logger.info(f"🎯 Boot {boat_idx} visiert Patch {best_idx} an (value = {best_val:.2f})")
                return
            else:
                logger.info(f"🛑 Boot {boat_idx}: 80% voll, kein passender Patch. Entleere.")
                self.resting_hours_left[boat_idx] = 20
                self.speed_factor[boat_idx] = 0.0
                return

        valid_candidates = [(i, val) for i, val in all_valid if self.can_pickup_patch(boat_idx, val, max_capacity)]
        if not valid_candidates:
            logger.info(f"🛑 Boot {boat_idx}: Kein Patch innerhalb Kapazität gefunden.")
            return

        best_idx, best_val = max(valid_candidates, key=lambda x: x[1])
        self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[best_idx]
        self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[best_idx]
        self.elements.target_patch_index[boat_idx] = best_idx
        logger.info(f"🎯 Boot {boat_idx} visiert Patch {best_idx} an (value = {best_val:.2f})")

    def assign_target_distance(self, boat_idx):
        if self.patches_model.num_elements_active() == 0:
            logger.info(f"⚓ Boot {boat_idx}: Keine Ziele mehr verfügbar.")
            return

        if self.elements.target_patch_index[boat_idx] != -1:
            return

        boat_lon = self.elements.lon[boat_idx]
        boat_lat = self.elements.lat[boat_idx]
        collected = self.elements.collected_value_current[boat_idx]
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

        if collected >= 0.8 * self.max_capacity:
            candidates.sort(key=lambda x: x[1])
            top_nearby = candidates[:5]
            for i, dist, patch_value in top_nearby:
                if self.can_pickup_patch(boat_idx, patch_value):
                    self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[i]
                    self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[i]
                    self.elements.target_patch_index[boat_idx] = i
                    logger.info(f"🎯 Boot {boat_idx} visiert Patch {i} an (Distanz = {dist:.4f}°)")
                    return
            if self.elements.resting_hours_left[boat_idx] == 0:
                logger.info(f"🛑 Boot {boat_idx}: 80% voll, aber kein naher passender Patch. Entleere.")
                self.elements.resting_hours_left[boat_idx] = self.resting_hours
                self.elements.speed_factor[boat_idx] = 0.0
            return

        valid_candidates = [c for c in candidates if self.can_pickup_patch(boat_idx, c[2])]
        if not valid_candidates:
            logger.info(f"🛑 Boot {boat_idx}: Kein erreichbarer Patch unter Kapazitätsgrenze verfügbar.")
            return

        best_idx, best_dist, _ = min(valid_candidates, key=lambda x: x[1])
        self.elements.target_lat[boat_idx] = self.patches_model.elements.lat[best_idx]
        self.elements.target_lon[boat_idx] = self.patches_model.elements.lon[best_idx]
        self.elements.target_patch_index[boat_idx] = best_idx
        logger.info(f"🎯 Boot {boat_idx} visiert Patch {best_idx} an (Distanz = {best_dist:.4f}°)")
