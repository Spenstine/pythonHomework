import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

migrationData = pd.read_csv('bird_tracking.csv', index_col = 0)


'''plotting migration routes for each bird'''
plt.figure(figsize = (10, 10))
for bird in migrationData.bird_name.unique():
    ind = migrationData.bird_name == bird
    y, x = migrationData.latitude[ind], migrationData.longitude[ind]
    plt.plot(x, y, '.', label = bird)

plt.legend(loc = 'lower right')
plt.title('Migration paths for the three observed birds'.title())
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('migration_path.png')

'''plotting 2D speed for Eric using plt.plot'''
ind = migrationData.bird_name == 'Eric'

#find if there is NA values in Eric's speed and removing them if so
speed = migrationData.speed_2d[ind]
if np.any(np.isnan(speed)):
    ind = ~np.isnan(migrationData.speed_2d[ind]) #'~' is bitwise not 
    speed = speed[ind]
plt.figure(figsize= (7, 7))
plt.xlabel('2D speed(m/s)')
plt.ylabel('frequency')
plt.title('Eric\'s 2D speed'.title())
#normed kwarg is depricated in matplotlib 3.3
#plt.hist(speed, bins = np.linspace(0, 30, 20), normed = True)
plt.hist(speed, bins = np.linspace(0, 30, 20), density = True, stacked = True)
plt.savefig('plt_hist.png')
plt.clf() #clear frame for future plots

'''plotting Eric's speed using pd.plot'''
ind = migrationData.bird_name == 'Eric'
migrationData.speed_2d[ind].plot(kind = 'hist', range=[0, 30])
plt.xlabel('2D speed(m/s)')
plt.ylabel('count')
plt.title('Eric\'s 2D speed'.title())
plt.savefig('pd_hist.png')

'''using dates to learn more about Eric's data'''
#add a timestamp column for datetime objects
#dates from date_time column are used, with the last 3 characters removed since they represent UTC offset
migrationData['timestamp'] = [datetime.datetime.strptime(migrationData.loc[i, 'date_time'][:-3], '%Y-%m-%d %H:%M:%S') for i in range(len(migrationData))]
times = migrationData.timestamp[ind]
elapsed_time = [time - times[0] for time in times]
elapsed_days = np.array(elapsed_time)/datetime.timedelta(days = 1)

#plot Eric's migration observations 
plt.figure()
plt.plot(elapsed_days)
plt.ylabel('Elapsed time (days)')
plt.xlabel('Observations')
plt.title('Eric\'s migration observations'.title())
plt.savefig('time_plot.png')

#Eric's daily average speed
next_day = 1
inds = []
dailySpeedMeans = []
for i, day in enumerate(elapsed_days):
    if day < next_day:
        inds.append(i)
    else:
        dailySpeedMeans.append(np.mean(migrationData.loc[inds, 'speed_2d']))
        next_day += 1
        inds = []

plt.figure()
plt.plot(dailySpeedMeans)
plt.ylabel('Daily Average Speed(m/s)')
plt.xlabel('Elapsed time(days)')
plt.title('Eric\'s daily mean speed'.title())
plt.savefig('daily_mean_speed.png')

'''plotting the map followed by birds using cartopy'''
proj = ccrs.Mercator()

plt.figure(figsize = (10, 10))
ax = plt.axes(projection = proj)
ax.set_extent((-25.0, 20.0, 52.0, 10.0))

#add land, ocean, coastline, and country borders
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle = ':')

bird_names = migrationData.bird_name.unique()
for name in bird_names:
    ix = migrationData['bird_name'] == name
    x, y = migrationData.longitude[ix], migrationData.latitude[ix]
    ax.plot(x, y, '.', transform = ccrs.Geodetic(), label = name)

plt.legend(loc = 'upper left')
plt.savefig('map.png')