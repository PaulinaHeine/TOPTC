import matplotlib
matplotlib.use('Qt5Agg')
from opendrift.models.basemodel import OpenDriftSimulation
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import logging
from opendrift.readers.reader_netCDF_CF_generic import Reader
from Modelle.animations import animation_custom
from Modelle.OpenDriftPlastCustom import OpenDriftPlastCustom
from Modelle.GreedyBoat_value import GreedyBoat_value
from Modelle.GreedyBoat_distance import GreedyBoat_distance

from collections import defaultdict
import random


# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datenpfad
data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024'

# Dataset laden
ds = xr.open_dataset(data_path)
print(ds)

# Plastikmodell initialisieren
o = OpenDriftPlastCustom(loglevel=logging.INFO)
r = Reader(data_path)
o.add_reader(r)

# boot initalisieren
b = GreedyBoat_distance(loglevel=logging.INFO, patches_model=o)
r = Reader(data_path)
b.add_reader(r)

lon_min = float(ds.longitude.min())
lon_max = float(ds.longitude.max())
lat_min = float(ds.latitude.min())
lat_max = float(ds.latitude.max())

o.simulation_extent = [lon_min, lon_max, lat_min, lat_max]
b.simulation_extent = [lon_min, lon_max, lat_min, lat_max]

# Startzeit vorbereiten
start_time = ds.time.values[0]
if not isinstance(start_time, datetime):
    start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)

# Geografisches Zentrum des Gebiets
mid_latitude = ds.latitude[int(len(ds.latitude) / 2)]
mid_longitude = ds.longitude[int(len(ds.longitude) / 2)]
depth = ds.depth.values[0]


steps = 500
dt = timedelta(hours=1)
o.time_step = dt
o.time_step_output = timedelta(hours=1)
o.time = start_time

b.time_step = dt
b.time_step_output = timedelta(hours=1)
b.time = start_time

o.seed_plastic_patch(radius_km = 4,number = 50, lon=mid_longitude, lat=mid_latitude, time = start_time, z = depth, seed = 1)
b.seed_boat(lon=mid_longitude, lat=mid_latitude,number=2, time = start_time, speed_factor=1) #ca 6kmh




o.prepare_run()
b.prepare_run()




for i in range(0,steps):
    print("Aktueller Simulationszeitpunkt:", o.time)
    b.update()
    o.update()


    # 4. Zeit voranschreiten
    o.time += dt
    b.time += dt
    i += 1




records = o.get_structured_history()
o.history = records
#print(o.history)

records_b = b.get_structured_history()
b.history = records_b
b.print_collection_summary()
#print(b.history)


animation_custom(model = o,fast = True, compare= b,size='value')


'''
# Aktive Patches anzeigen
print("\n=== Aktive Patches ===")
for i in range(o.num_elements_active()):
    val = o.elements.value[i]
    wgt = o.elements.weight[i]
    area = o.elements.area[i]
    dens = o.elements.density[i]
    print(f"Patch {i}: Wert = {val:.2f}, Gewicht = {wgt:.2f} kg, Fläche = {area:.2f} m², Dichte = {dens:.3f} kg/m³")

# Deaktivierte Patches anzeigen
print("\n=== Deaktivierte Patches ===")
for i in range(o.num_elements_deactivated()):
    val = o.elements.value[i]
    wgt = o.elements.weight[i]
    area = o.elements.area[i]
    dens = o.elements.density[i]
    print(f"[DEAKTIVIERT] Patch {i}: Wert = {val:.2f}, Gewicht = {wgt:.2f} kg, Fläche = {area:.2f} m², Dichte = {dens:.3f} kg/m³")

#print(o.times)
'''




# Animation anzeigen

# TODO value counter
# todo routenfindungchecken, euclidische distanz? -> ja
# Todo notebooks ordnen
# todo: zeit und value messung
# todo: Instances vorbereiten -> Big, medium, small (wieviele?)
# todo in der grafik verschwinden zuerst kleinere punkte obwohl zuerst die mit dem höchsten value eingefangen werden sollen





# todo: hohe dichtebereiche
# todo: richtung in die sie treiben?
