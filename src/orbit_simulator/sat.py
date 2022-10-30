from skyfield.api import load
from ground_path import ground_path
from datetime import datetime

number = "00900"
resource_url = 'https://celestrak.org/NORAD/elements/gp.php?CATNR=' + number

fname = 'tle_hodoyoshi_1.txt'
satellites = load.tle(resource_url, filename=fname, reload=True)

f = open(fname,"r")
satName = f.readline().strip()
f.close
satellite = satellites[satName]

date_from = datetime(2022,1,1,0,0,0,0)
date_to =   datetime(2022,1,1,10,0,0,0)

freq = 5 # generate data point every x seconds
df = ground_path(satellite, date_from, date_to, freq)

df.to_csv(number + '.csv')
