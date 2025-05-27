import matplotlib
matplotlib.use('Qt5Agg')
from datetime import datetime, timedelta
import xarray as xr
from opendrift.readers.reader_netCDF_CF_generic import Reader
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import logging
from Modelle.OpenDriftPlastCustom_RUN import OpenDriftPlastCustom
import random

random.seed(42)
np.random.seed(42)


# Datenpfad
data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024'

# Laden des Datensatzes mit xarray
ds = xr.open_dataset(data_path)
print(ds)

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Einrichten des OceanDrift-Modells
o = OpenDriftPlastCustom(loglevel=logging.INFO) # TODO nur die loggerinfo und das von loglevel20


r = Reader(data_path)

o.add_reader(r)




# Sicherstellen, dass start_time ein datetime-Objekt ist
start_time = ds.time.values[0]
if not isinstance(start_time, datetime):
    start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)

# Setzen der Partikel
mid_latitude = ds.latitude[int(len(ds.latitude) / 2)]
mid_longitude = ds.longitude[int(len(ds.longitude) / 2)]
depth = ds.depth.values[0]  # Die einzige verfügbare Tiefe


o.seed_plastic_patch(radius_km = 1,number = 20, lon=mid_longitude, lat=mid_latitude, time = start_time, z = depth)
print("=== Status vor Simulation ===")
print("Total:", o.num_elements_total())
print("Active:", o.num_elements_active())
print("Deactivated:", o.num_elements_deactivated())

# Simulation durchführen
o.run(duration=timedelta(hours=100))

for i in range(o.num_elements_active()):
    val = o.elements.value[i]
    wgt = o.elements.weight[i]
    area = o.elements.area[i]
    dens = o.elements.density[i]

    print(f"Patch {i}: Wert = {val:.2f}, Gewicht = {wgt:.2f} kg, Fläche = {area:.2f} m², Dichte = {dens:.3f} kg/m³")

for i in range(o.num_elements_deactivated()):
    val = o.elements.value[i]
    wgt = o.elements.weight[i]
    area = o.elements.area[i]
    dens = o.elements.density[i]

print(f"Patch {i}: Wert = {val:.2f}, Gewicht = {wgt:.2f} kg, Fläche = {area:.2f} m², Dichte = {dens:.3f} kg/m³")



print(o.export_variables)
print(o.history)

print(type(o.history))
print(type(o.history[0]))
print(type(o.history[0][0]))

times = o.get_time_array()[0]
print(times)




o.animation(fast = True, color='current_drift_factor')


#o.animation_custom(fast = True, color='current_drift_factor')



