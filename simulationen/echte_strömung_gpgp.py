import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime, timedelta
import xarray as xr
from opendrift.readers.reader_netCDF_CF_generic import Reader
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.plastdrift import PlastDrift # Beispiel, ändere den Namen entsprechend der tatsächlichen Verfügbarkeit


# Datenpfad
data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024'

# Laden des Datensatzes mit xarray
ds = xr.open_dataset(data_path)
print(ds)

# Einrichten des OceanDrift-Modells
o = PlastDrift(loglevel=20)
r = Reader(data_path)
o.add_reader(r)
o.set_config('environment:fallback:land_binary_mask', 0)
#o.set_config('general:use_auto_landmask', False)  # Vermeidung automatischer Landmasken, falls vorhanden

# Setzen der Partikel an einer repräsentativen Position und Tiefe innerhalb der gegebenen Grenzen
mid_latitude = ds.latitude[int(len(ds.latitude) / 2)]
mid_longitude = ds.longitude[int(len(ds.longitude) / 2)]
depth = ds.depth.values[0]  # Die einzige verfügbare Tiefe

# Startzeit aus den Daten wählen
start_time = ds.time.values[0]
if not isinstance(start_time, datetime):
    start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)

# Partikel freisetzen
o.seed_elements(lon=mid_longitude, lat=mid_latitude, number=1000, radius=40000, z=-depth, time=start_time)

# Simulation durchführen
o.run(duration=timedelta(hours=193), time_step=timedelta(hours=1), time_step_output=timedelta(hours=1))

# Ergebnis visualisieren
o.animation(fast=True, skip_frames=0, time_step=timedelta(hours=1), time_step_output=timedelta(hours=1))
