import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM

def normalizer(someseries):
    maxval = max(someseries)
    minval = min(someseries)
    newseries = (someseries - minval)/(maxval-minval)
    return newseries


def standardizer(someseries):
    mean = np.mean(someseries)
    stddev = np.std(someseries)
    someseries = (someseries - mean)/stddev

    return someseries


df = pd.read_csv('last_impl_datset.csv')
df = df.fillna("Nil")
df = df[df['Fog node'] != "Nil"]
#df = df[df['Distance'] != "undefined"]
df = df[df['Fog node'] != "test"]
temp = (df['Cost'].astype(float))
temp2 = standardizer(df['Distance'].astype(float))
df = df.drop(columns = ['Device','Time', 'Unnamed: 0', 'Day', 'Cost', 'Distance'])
df['Distance']  = temp2
df['Cost'] = np.round_(normalizer(temp), 2)
df = df.drop(columns = ['xcell', 'ycell'])
cols = list(df.columns)
cols = cols[2:len(cols)] + cols[0:2]
df = df[cols]

print("Shape of the dataframe is {}".format(df.shape))

prevvals = 10

inputrow = []
outputrow = []
outputcoordinates = []


# this part makes an input vector for the lstm
# it take a window of the previous values (prevvals)
# the shape of the input df is samples * steps * features like I mentioned
for counter in range(prevvals, len(df) -prevvals- 1):
    temp = df.iloc[counter:counter+prevvals, 0:len(cols)-2]
    inputrow.append(np.array(temp))
    outputrow.append(df.iloc[counter+prevvals+1, len(cols)-3:len(df.columns)])


inputdf = np.array(inputrow)
outputdf = np.array(outputrow)

cutoff = int(0.80 * len(inputdf))

X_train = inputdf[0:cutoff]
X_test = inputdf[cutoff:len(inputdf)]
y_train = outputdf[0:cutoff]
y_test = outputdf[cutoff:len(outputdf)]


model = Sequential()
model.add(LSTM(50, input_shape=(inputdf.shape[1], inputdf.shape[2]), return_sequences = True))
model.add(LSTM(50))
model.add(Dense(10,  activation = 'sigmoid'))
model.add(Dense(1))
model.compile(loss="mean_absolute_error", optimizer='adam')
model.fit(X_train, y_train[:,0], epochs=1, batch_size=5000, verbose=True, validation_data = [X_test, y_test[:,0]])

result = model.predict(X_test)
# just compare the result to the y_test visually to verify if it actually follows the trend


model.save('costpredictor_forexperiments.h5')
