
import matplotlib
from scipy.cluster.hierarchy import weighted
from opendrift.models.basemodel import OpenDriftSimulation
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import logging
from opendrift.readers.reader_netCDF_CF_generic import Reader
from visualisations.animations import animation_custom
from Modelle.OpenDriftPlastCustom import OpenDriftPlastCustom
from Modelle.GreedyBoat import GreedyBoat

from collections import defaultdict
import random
matplotlib.use('Qt5Agg')


def run_genetic_initial( time_frame = 100, plastic_radius = 10, plastic_number = 500, plastic_seed = 1, boat_number = 2, speed_factor_boat=3, animation = False, weighted_alpha_value = 0.5 ):#, max_capacity_value=6000, resting_hours_amount=12

    # Initiate

    # Logging konfigurieren
    logging.basicConfig(level=logging.CRITICAL)#info
    logging.disable(logging.CRITICAL)
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
    # 2. num_boats (integer)
    num_boats_value = 2  # Example: You want to simulate 5 boats

    # 3. num_patches (integer)
    # This usually comes from the patches_model.
    num_patches_value = plastic_number  # Get the total number of patches from your patches_model 'o'

    # 4. patches (object representing patch data)
    # This could be the raw elements array from your patches_model, or a processed version.
    patches_data = o.elements  # Using the elements from your patches_model 'o'

    # 5. current_data (object representing ocean current data)
    # This would typically be a reader or an object that provides current data.
    # For a minimal example, we'll use a placeholder.
    # In a real scenario, this would be an OpenDrift Reader object or similar.
    current_data_source = None  # Placeholder: Replace with your actual current data reader/object
    # Example: from opendrift.readers import reader_netcdf_CF
    # current_data_source = reader_netcdf_CF.Reader(...)

    # 6. weighted_alpha (float, optional, has a default)
    weighted_alpha_value = 0.7  # Example value, or let it use the default 0.5

    # 7. start_position (tuple, optional, has a default)
    # If you want to use the default (-133.5, 24.5), you don't need to pass it explicitly.
    # Otherwise, specify it:
    # start_pos = (-133.0, 25.0)

    # Now, the valid initialization:
    print("Attempting to initialize GeneticBoat...")
    b = GreedyBoat(
        patches_model=o,  # Your patches model instance
        num_boats=num_boats_value,
        num_patches=num_patches_value,
        patches=patches_data,
        current_data=current_data_source,
        weighted_alpha=weighted_alpha_value,
        # start_position=start_pos, # Optional, use default if commented out
        time_step=timedelta(minutes=10)  # Example: Pass OpenDriftSimulation base args
    )

    print("GeneticBoat initialized successfully!")

    b = GreedyBoat( patches_model=o, weighted_alpha = weighted_alpha_value) # , max_capacity= max_capacity_value, resting_hours= resting_hours_amount
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
    #custom_history = o.custom_history_list
    #print(o.history)
    #print(custom_history)
    b.history = b.get_structured_history()     ### Das ist was ich brauche als initial startlösung

    print(b.history)
    short_logbook = b.print_collection_summary() # infos über die boote insgesammt


    if animation == True:
        animation_custom(model = o,fast = True, compare= b,size='value', show_trajectories=False)
        b.plot(fast = True, show_trajectories=True) # zeigt die routen der boote an

    return b.history, short_logbook


logbook = []
history = []

alpha = [0,0.25,0.5,0.75,1]
alpha = [0]
for i in alpha:
    print("alpha:", i)
    hist, loo = run_genetic_initial(
             time_frame = 500,
             plastic_radius = 10,
             plastic_number = 1000,
             plastic_seed = 1000,
             boat_number = 2,
             speed_factor_boat = 3,
             weighted_alpha_value = i, # 0 Value -> 1 Distanz
             animation = False)
    logbook.append(loo) # im prinzip fitness
    history.append(hist)



#print(loo)
#print(history)


