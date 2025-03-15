#Daten aus https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/download?dataset=cmems_mod_glo_phy_anfc_0.083deg_PT1H-m_202406

import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime, timedelta
import xarray as xr
from opendrift.readers.reader_netCDF_CF_generic import Reader
from opendrift.models.oceandrift import OceanDrift
from opendrift.models.plastdrift import PlastDrift # Beispiel, ändere den Namen entsprechend der tatsächlichen Verfügbarkeit
import numpy as np
np.random.seed(42)

# Datenpfad
data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/currency_data/gpgp_long'

# Laden des Datensatzes mit xarray
ds = xr.open_dataset(data_path)
print(ds)

# Einrichten des OceanDrift-Modells
o = PlastDrift(loglevel=20)
o2 = PlastDrift(loglevel=20)
r = Reader(data_path)
o.add_reader(r)
o2.add_reader(r)


# Setzen der Partikel an einer repräsentativen Position und Tiefe innerhalb der gegebenen Grenzen
mid_latitude = ds.latitude[int(len(ds.latitude) / 2)]
mid_longitude = ds.longitude[int(len(ds.longitude) / 2)]
depth = ds.depth.values[0]  # Die einzige verfügbare Tiefe

# Startzeit aus den Daten wählen
start_time = ds.time.values[0]
if not isinstance(start_time, datetime):
    start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)

# Partikel freisetzen
o.seed_elements(lon=mid_longitude, lat=mid_latitude, number=1, radius=0, z=-depth, time=start_time,terminal_velocity=0.00005, wind_drift_factor=0.01)
o.run(duration=timedelta(hours=193))


# Second run schwere partikel
o2.seed_elements(lon=mid_longitude, lat=mid_latitude, number=1, radius=0, z=-depth, time=start_time,terminal_velocity=0.5, wind_drift_factor=0.05)
o2.run(duration=timedelta(hours=193))




# Ergebnis visualisieren
o.animation(compare=o2, fast=True,
            legend=['light', 'heavy'])