import matplotlib
matplotlib.use('Qt5Agg')
from datetime import datetime, timedelta
import xarray as xr
from opendrift.readers.reader_netCDF_CF_generic import Reader
from opendrift.models.oceandrift import OceanDrift
from Modelle.Plastic_Model_Custom import PlastElement, PlastDrift_M
from Modelle.oceandrift_custom import OpenDriftPlastCustom

# Datenpfad
data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024'

# Laden des Datensatzes mit xarray
ds = xr.open_dataset(data_path)
print(ds)

# Einrichten des OceanDrift-Modells
o = OpenDriftPlastCustom(loglevel=20)
o2 = OpenDriftPlastCustom(loglevel=20)

r = Reader(data_path)

o.add_reader(r)
#o.set_config('environment:fallback:land_binary_mask', 0)

o2.add_reader(r)
#o2.set_config('environment:fallback:land_binary_mask', 0)

# Sicherstellen, dass start_time ein datetime-Objekt ist
start_time = ds.time.values[0]
if not isinstance(start_time, datetime):
    start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)

# Setzen der Partikel
mid_latitude = ds.latitude[int(len(ds.latitude) / 2)]
mid_longitude = ds.longitude[int(len(ds.longitude) / 2)]
depth = ds.depth.values[0]  # Die einzige verfügbare Tiefe

o.seed_elements(lon=mid_longitude, lat=mid_latitude, number=20, radius=30000, z=-depth, time=start_time)

# Simulation durchführen
o.run(duration=timedelta(hours=390))

#o2.seed_elements(lon=mid_longitude, lat=mid_latitude, number=10, radius=30000, z=-depth, time=start_time)

o2.seed_elements(lon=mid_longitude, lat=mid_latitude, number=20, radius=30000,time=start_time, z=-depth )#, weight=0.025, size=0.0025, density=250)


# Simulation durchführen
o2.run(duration=timedelta(hours=390))

# Ergebnis visualisieren
o.animation(fast = True, compare = o2)
