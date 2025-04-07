import matplotlib
matplotlib.use('Qt5Agg')
from datetime import datetime, timedelta
import xarray as xr
from opendrift.readers.reader_netCDF_CF_generic import Reader


from Modelle.OpenDriftPlastCustom import OpenDriftPlastCustom

# Datenpfad
data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024'

# Laden des Datensatzes mit xarray
ds = xr.open_dataset(data_path)
print(ds)

# Einrichten des OceanDrift-Modells
o = OpenDriftPlastCustom(loglevel=20)


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

#o.seed_elements(lon=mid_longitude, lat=mid_latitude, number=20, radius=30000, z=-depth, time=start_time)
o.seed_plastic_patch(radius_km = 200,number = 400, lon=mid_longitude, lat=mid_latitude, time = start_time)


# Simulation durchführen
o.run(duration=timedelta(hours=190))


# Ergebnis visualisieren
o.animation(fast = True, color='current_drift_factor')
