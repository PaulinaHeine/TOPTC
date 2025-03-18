# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

# MODIFIED BY PAULINA HEINE

import numpy as np
import logging
from opendrift.models.oceandrift import OceanDrift, Lagrangian3DArray

logger = logging.getLogger(__name__)

class PlastElement(Lagrangian3DArray):
    variables = Lagrangian3DArray.add_variables([
        ('weight', {'dtype': np.float32, 'units': 'kg', 'default': 0.1,
                    'description': 'Weight of the particle, affecting its inertia'}),
        ('size', {'dtype': np.float32, 'units': 'm²', 'default': 0.1,
                  'description': 'Physical size of the particle which affects drag'}),
        ('density', {'dtype': np.float32, 'units': 'kg/m³', 'default': 250,
                     'description': 'Density of the particle, affecting its buoyancy'})
    ])

class PlastDrift_M(OceanDrift):
    ElementType = PlastElement

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'sea_surface_wave_stokes_drift_x_velocity': {'fallback': 0},
        'sea_surface_wave_stokes_drift_y_velocity': {'fallback': 0},
        'sea_floor_depth_below_sea_level': {'fallback': 10000},
        'land_binary_mask': {'fallback': None},
    }

    def __init__(self, *args, **kwargs):
        super(PlastDrift_M, self).__init__(*args, **kwargs)

    def update(self):
        """Update positions and properties of elements based on weight, size, and density."""
        self.advect_ocean_current()

        # Update horizontal positions based on the drag influence from size and the inertia from weight
        for i in range(self.num_elements_active()):
            drag_influence = self.calculate_drag(self.elements.size[i])
            inertia_influence = self.calculate_inertia(self.elements.weight[i], self.elements.size[i])
            self.elements.lon[i] += drag_influence * inertia_influence * self.environment.x_sea_water_velocity[i] * self.time_step.total_seconds()
            self.elements.lat[i] += drag_influence * inertia_influence * self.environment.y_sea_water_velocity[i] * self.time_step.total_seconds()

    def calculate_drag(self, size):
        """Calculate the drag influence based on particle size."""
        return size * 0.1  # Example scaling factor for drag

    def calculate_inertia(self, weight, size):
        """Calculate the inertia influence based on weight and size."""
        return weight / size  # Simple ratio of weight to size as inertia factor
