


import matplotlib
matplotlib.use('TkAgg')
from datetime import datetime, timedelta
import numpy as np
from opendrift.readers.reader_constant_2d import Reader
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic

lon, lat = np.meshgrid(np.linspace(2,6,30), np.linspace(59,62,30))
lon0 = 4
lat0 = 60.5

lon = np.linspace(0,5,30)
lat = np.linspace(59,62,30)

r = reader_netCDF_CF_generic.Reader(
    'https://thredds.met.no/thredds/dodsC/sea/norkyst800m/1h/aggregate_be')

#r = reader_netCDF_CF_generic.Reader('norkyst800_16Nov2015.nc')

o = OceanDrift(loglevel=20)
o.set_config('environment:fallback:land_binary_mask', 0)
o.add_reader(r)
o.seed_elements(lon=3, lat=60.5, number=1000, radius=30000, time=datetime(2024, 6, 15, 1, 11, 11 ))
o.run(duration=timedelta(hours=11))
o.animation(fast=True)
