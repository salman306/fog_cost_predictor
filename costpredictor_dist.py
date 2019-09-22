
#the base cost predictor with the distance as an input. Has the post quantizer

import pandas as pd
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers.merge import concatenate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from math import radians, sin, cos, acos
from geopy import distance
from sklearn.svm import SVR
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras import optimizers
from sklearn.neighbors import KNeighborsRegressor

def normalizer(someseries):
    maxval = max(someseries)
    minval = min(someseries)
    newseries = (someseries - minval)/(maxval-minval)
    return newseries

def distancecalc(data, fogdata):
    fognode = data['Fog node']
    fognodedata = np.array(fogdata.loc[fognode - 1])

    foglong = list(fognodedata[:,1])
    foglat = list(fognodedata[:,2])
    devlat = list(data['Lat'])
    devlong = list(data['Long'])

    distance_list = []
    for counter in range(0, data.shape[0]):
        distance_list.append(distance.distance((foglat[counter],foglong[counter]),(devlat[counter], devlong[counter])).meters)
    return (np.array(distance_list))

def latlongconv_fog(data, fogdata):

    fognode = data['Fog node']
    fognodedata = np.array(fogdata.loc[fognode - 1])


    longitude = np.array(fognodedata[:,1])
    latitude = np.array(fognodedata[:,2])

    x_cartesian_fog = normalizer(np.cos(np.radians(longitude)) + np.cos(np.radians(latitude)))
    y_cartesian_fog = normalizer(np.cos(np.radians(longitude)) + np.sin(np.radians(latitude)))
    z_cartesian_fog = normalizer(np.sin(np.radians(latitude)))

    x_cart_dev = data['x_cart']
    y_cart_dev = data['y_cart']
    z_cart_dev = data['z_cart']

    distance = np.sqrt(((x_cartesian_fog - x_cart_dev)**2) + ((y_cartesian_fog - y_cart_dev)**2) + ((z_cartesian_fog - z_cart_dev)**2))

    return distance

def quantizer (levels, prediction, groundtruth, maxval, minval):

    margins = np.linspace(minval, maxval, levels)

    prediction_quant = []
    for value in prediction:
        tempcount = 0
        for margin in margins:
            if (value > margin):
                tempcount = tempcount + 1
            else:
                prediction_quant.append(tempcount)
                break


    groundtruth_quant = []
    for value in groundtruth:
        tempcount = 0
        for margin in margins:

            if (value > margin):
                tempcount = tempcount + 1
            else:
                groundtruth_quant.append(tempcount)
                break

    prediction_quant = np.array(prediction_quant)
    groundtruth_quant = np.array(groundtruth_quant)

    acc = 100 * sum(prediction_quant == groundtruth_quant) / len(prediction_quant)
    print(acc)
    return (prediction_quant, groundtruth_quant, acc)

def standardizer(someseries):
    mean = np.mean(someseries)
    stddev = np.std(someseries)
    someseries = (someseries - mean)/stddev

    return someseries

df = pd.read_csv('~/Desktop/Common/Project/Project_pred/Datasets/d125r_2_csv_tracetest_pp.csv')
df = df.fillna("Nil")
df = df[df['Fog node'] != "Nil"]
#df = df[df['Distance'] != "undefined"]
#df = df[df['Fog node'] != "test"]
#df = df[df['Cost'] != "test"]
temp = (df['Cost'].astype(float))
temp2 = standardizer(df['Distance'].astype(float))
df = df.drop(columns = ['Device','Time', 'Unnamed: 0', 'Day', 'Cost', 'Distance', 'Test check'])
df['Distance']  = temp2
df['Cost'] = np.round_(normalizer(temp), 2)

df = df.drop(columns = ['Long', 'Lat', 'xcell', 'ycell'])

inputdf = df.iloc[:,0:df.shape[1]-1]
outputdf = df.iloc[:, df.shape[1]-1]
print(inputdf.columns)

X_train, X_test, y_train, y_test = train_test_split(inputdf, outputdf, test_size=0.15, random_state=11)
y_train = np.array(y_train).reshape(y_train.shape[0],1)
y_test = np.array(y_test).reshape(y_test.shape[0],1)


model2 = Sequential()
model2.add(Dense(100, input_dim = inputdf.shape[1], activation = 'sigmoid'))
model2.add(Dense(100, activation = 'sigmoid'))
model2.add(Dense(1, activation =  "linear"))
#model.add(Dropout(0.5))
model2.compile(optimizer = 'adam', loss = 'mae')
model2.fit(X_train, y_train, epochs = 1000, batch_size = 5000, verbose = 2, validation_data = [X_test, y_test], callbacks = [])

prediction = model2.predict(X_test)

final = quantizer(4, prediction, y_test, max(outputdf), min(outputdf))
final = quantizer(5, prediction, y_test, max(outputdf), min(outputdf))
final = quantizer(6, prediction, y_test, max(outputdf), min(outputdf))
print(np.unique(final[0], return_counts = True))
print(np.unique(final[1], return_counts = True))

plt.plot(y_test, 'r', label = 'groundtruth')
plt.plot(np.array(prediction), 'b', label = 'prediction')
plt.legend()
plt.show()


prediction = model2.predict(X_train)
final = quantizer(6, prediction, y_train, max(outputdf), min(outputdf))
print(np.unique(final[0], return_counts = True))
print(np.unique(final[1], return_counts = True))


prediction = model2.predict(X_test)
final = quantizer(6, prediction, y_test, max(outputdf), min(outputdf))
print(np.unique(final[0], return_counts = True))
print(np.unique(final[1], return_counts = True))

error2 = mean_absolute_error(y_test, prediction)
print(error2)


svm = SVR(C=1, epsilon=.1)
svm.fit(X_train, y_train)

prediction = svm.predict(X_test)
prediction = np.array(prediction)
error2_svm = mean_absolute_error(y_test, prediction)
print("SVM error:", error2_svm)

prediction = model2.predict(X_train)
plt.plot(y_train, 'r')
plt.plot(np.array(prediction), 'b')
plt.show()


for noofneigh in range(1, 15):
    neigh = KNeighborsRegressor(n_neighbors=noofneigh)
    neigh.fit(X_train, y_train)
    prediction = neigh.predict(X_test)

    print(str(noofneigh) + "neighbors: " + str(mean_absolute_error(prediction, y_test)))



