import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime, timedelta
import xarray as xr
from opendrift.readers.reader_netCDF_CF_generic import Reader
from opendrift.models.oceandrift import OceanDrift

# Datenpfad
data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/currency_data/Nordirgendwas'

# Laden des Datensatzes mit xarray
ds = xr.open_dataset(data_path)
print(ds)

# Einrichten des OceanDrift-Modells
o = OceanDrift(loglevel=20)
r = Reader(data_path)
o.add_reader(r)
o.set_config('environment:fallback:land_binary_mask', 0)

# Sicherstellen, dass start_time ein datetime-Objekt ist
start_time = ds.time.values[0]
if not isinstance(start_time, datetime):
    start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)

# Setzen der Partikel
mid_latitude = ds.latitude[int(len(ds.latitude) / 2)]
mid_longitude = ds.longitude[int(len(ds.longitude) / 2)]
depth = ds.depth.values[0]  # Die einzige verfügbare Tiefe

o.seed_elements(lon=mid_longitude, lat=mid_latitude, number=1000, radius=30000, z=-depth, time=start_time)

# Simulation durchführen
o.run(duration=timedelta(hours=21))

# Ergebnis visualisieren
o.animation(fast=True)
