import matplotlib

matplotlib.use('Qt5Agg')
from datetime import datetime, timedelta
import xarray as xr
from opendrift.readers.reader_netCDF_CF_generic import Reader
from Modelle.OpenDriftPlastCustom import OpenDriftPlastCustom
from Modelle.OpenDriftBoat import OpenDriftBoat  # <- das neue Bootmodell

# Daten laden
data_path = '/Users/paulinaheine/Master Business Analytics/Masterarbeit/Technisches/TOPTC/data/currency_data/current_june2024'

ds = xr.open_dataset(data_path)

r = Reader(data_path)

# Plastikmodell
o_plast = OpenDriftPlastCustom(loglevel=20)
o_plast.add_reader(r)

# Bootmodell
o_boot = OpenDriftBoat(loglevel=20)
o_boot.add_reader(r)

# Startzeit ermitteln
start_time = ds.time.values[0]
if not isinstance(start_time, datetime):
    start_time = datetime.utcfromtimestamp(start_time.astype(int) * 1e-9)

# Seed Plastik
mid_latitude = ds.latitude[int(len(ds.latitude) / 2)]
mid_longitude = ds.longitude[int(len(ds.longitude) / 2)]
o_plast.seed_plastic_patch(lon=mid_longitude, lat=mid_latitude, time=start_time, number=100, radius_km=30)

# Seed Boot
o_boot.seed_boat(lon=mid_longitude + 0.5, lat=mid_latitude + 0.5, time=start_time)

# Schrittweise Simulation (manuell)
duration = timedelta(hours=24)
dt = timedelta(minutes=30)
steps = int(duration.total_seconds() / dt.total_seconds())

for _ in range(steps):
    # Schrittweise simulieren
    o_plast.step()

    # Aktuelle Plastikpositionen holen
    patch_lons = o_plast.elements.lon
    patch_lats = o_plast.elements.lat

    # Nur aktive Patches
    valid = ~np.isnan(patch_lons)
    if np.any(valid):
        # Boot zielt auf nächsten Patch
        boat_lon = o_boot.elements.lon[0]
        boat_lat = o_boot.elements.lat[0]
        dists = np.sqrt((patch_lons[valid] - boat_lon) ** 2 + (patch_lats[valid] - boat_lat) ** 2)
        nearest = np.argmin(dists)
        target_lon = patch_lons[valid][nearest]
        target_lat = patch_lats[valid][nearest]
        o_boot.set_target(target_lon, target_lat)

    # Boot weiter bewegen
    o_boot.step()
