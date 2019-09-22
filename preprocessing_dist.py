
#same as original preprocess.py

import pandas as pd
import numpy as np
import geopy.distance
from math import radians
#print geopy.distance.vincenty(coords_1, coords_2).km


def gridpoints(data):
    xcord = data['Long']
    ycord = data['Lat']

    #start point of the grid
    minx = min(xcord)
    maxy = max(ycord)

    #end point of the grid
    maxx = max(xcord)
    miny = min(ycord)


    return ((minx, maxy), (maxx, miny))

def gridintervals (startpoints, endpoints):
    x_start = startpoints[0]
    x_end = endpoints[0]
    stepsize_x = (x_end - x_start)/gridsize[0]
    x_intervals = np.arange(x_start,x_end, stepsize_x)

    y_start = startpoints[1]
    y_end = endpoints[1]
    stepsize_y = (y_start - y_end)/gridsize[1]
    y_intervals = np.arange(y_end, y_start, stepsize_y)

    x_intervals = np.append(x_intervals, endpoints[0])
    y_intervals = np.append(y_intervals, start[0])

    return (x_intervals, y_intervals)

def cellassignment (data, intervals):
    xcord = data['Long']
    ycord = data['Lat']

    xgridcord = []
    xintervals = intervals[0]
    for xcoordinate in xcord:
        tempcount = 0
        for xintelement in xintervals:
            if xcoordinate > xintelement:
                tempcount = tempcount + 1
            else:
                xgridcord.append(tempcount)
                break

    ygridcord = []
    yintervals = intervals[1]
    for ycoordinate in ycord:
        tempcount = 0
        for yintelement in yintervals:
            if ycoordinate > yintelement:
                tempcount = tempcount + 1
            else:
                ygridcord.append(tempcount)
                break

    return (xgridcord,ygridcord)


def gridmapper(xcell, ycell, gridsize):
    cellids = []

    for counter in range(0, len(xcell)):
        cellids.append(xcell[counter] + (ycell[counter] + 1) * gridsize[0])

    return cellids

def looper(someseries, somestring):
    result = []

    if somestring == "day":
        for items in someseries:
            result.append(items.dayofweek)
        return pd.Series(result)
    elif somestring == "minute":
        for items in someseries:
            result.append(items.minute)
        return pd.Series(result)
    elif somestring == "second":
        for items in someseries:
            result.append(items.second)
        return pd.Series(result)
    elif somestring == "hour":
        for items in someseries:
            result.append(items.hour)
        return pd.Series(result)


def latlongconv(data):
    longitude = data['Long']
    latitude = data['Lat']

    x_cartesian = normalizer(np.cos(np.radians(longitude)) + np.cos(np.radians(latitude)))
    y_cartesian = normalizer(np.cos(np.radians(longitude)) + np.sin(np.radians(latitude)))


    data['x_cart'] = x_cartesian
    data['y_cart'] = y_cartesian

    return data

def normalizer(someseries):
    maxval = max(someseries)
    minval = min(someseries)
    newseries = (someseries - minval)/(maxval-minval)
    return newseries


df = pd.read_csv("o8_csv.csv")
gridsize = (10,10) # number of xblocks and yblocks
start = gridpoints(df)[0]
end = gridpoints(df)[1]
intervals = gridintervals(start, end)
(xcell, ycell) = cellassignment(df, intervals)
cellids = gridmapper(xcell, ycell, gridsize)
df['CellID'] = cellids
df = latlongconv(df)



td = df['Time']
td = pd.to_datetime(td)
df['Day'] = looper(td, "day")
time_seconds = looper(td, "hour") * 3600 +  looper(td, "minute") *60 + looper(td, "second")
df['sin_time'] = np.sin(time_seconds)
df['cos_time'] = np.cos(time_seconds)

df.to_csv("o8_processed.csv")
