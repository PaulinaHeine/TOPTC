import matplotlib
from scipy.cluster.hierarchy import weighted

matplotlib.use('Qt5Agg')
from opendrift.models.basemodel import OpenDriftSimulation
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import logging
from opendrift.readers.reader_netCDF_CF_generic import Reader
from Modelle.animations import animation_custom
from Modelle.OpenDriftPlastCustom import OpenDriftPlastCustom
from Modelle.GreedyBoat import GreedyBoat
from collections import defaultdict
import random


def main(boat_target_mode = None,time_frame = None, plastic_radius = None, plastic_number = None, plastic_seed = None, boat_number = None, speed_factor_boat=1, animation = False, weighted_alpha_value = 0.5 ):#, max_capacity_value=6000, resting_hours_amount=12

    # Initiate

    # Logging konfigurieren
    logging.basicConfig(level=logging.CRITICAL)#info
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
    b = GreedyBoat(loglevel=logging.INFO, patches_model=o, target_mode=boat_target_mode, weighted_alpha = weighted_alpha_value) # , max_capacity= max_capacity_value, resting_hours= resting_hours_amount


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


    steps = time_frame
    dt = timedelta(hours=1)
    o.time_step = dt
    o.time_step_output = timedelta(hours=1)
    o.time = start_time

    b.time_step = dt
    b.time_step_output = timedelta(hours=1)
    b.time = start_time

    o.seed_plastic_patch(radius_km = plastic_radius,number = plastic_number, lon=mid_longitude, lat=mid_latitude, time = start_time, z = depth, seed = plastic_seed)
    b.seed_boat(lon=mid_longitude, lat=mid_latitude,number=boat_number, time = start_time, speed_factor= speed_factor_boat) #ca 6kmh
    print("🚤 Speed-Faktoren nach Seeding:", b.elements.speed_factor[:b.num_elements_active()])

    o.prepare_run()
    b.prepare_run()



    r = 0
    for i in range(0,steps):
        print("Aktueller Simulationszeitpunkt:", o.time)
        b.update()
        o.update()
        r += 1
        #if r % 50 == 0:
            #o.seed_random_edge_patches(seed = plastic_seed)


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

    if animation == True:
        animation_custom(model = o,fast = True, compare= b,size='value', show_trajectories=False)
        #b.plot(fast = True, show_trajectories=True) # zeigt die routen der boote an



main(boat_target_mode = "weighted",  # value or distance or weighted
     time_frame = 200,
     plastic_radius = 10,
     plastic_number = 150,
     plastic_seed = 2,
     boat_number = 2,
     speed_factor_boat = 3,
     weighted_alpha_value =0.9,
     animation = True)
     #max_capacity_value = 50000,
     #resting_hours_amount = 12)





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

# TODO value counter updaten
# Todo notebooks ordnen


