#multiday fog predictor with test set from the last day only

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


df = pd.read_csv('trace_133_8_preprocessed.csv')
df = df.fillna("Nil")
#df = df[df['Fog node'] != "Nil"]
#df = df[df['Distance'] != "undefined"]
temp = df['Cost'].astype(float)
df = df.drop(columns = ['Long', 'Lat', 'Device','Time', 'Unnamed: 0', 'Cost', 'Distance'])
df = pd.get_dummies(df, columns = ['Fog node'])
df['Cost'] = temp
days = np.unique(np.array(df['Day']))

traindf = pd.DataFrame()
for counter in range(0, len(days) -1):
    traindf = pd.concat([traindf, df[df['Day'] == days[counter]]])

testdf = df[df['Day'] == days[-1]]
df = traindf


inputfeatures = 8
inputdf1 = df.iloc[:,0:inputfeatures]
outputdf1 = df.iloc[:, inputfeatures:df.shape[1]-1]
inputdf2 = testdf.iloc[:,0:inputfeatures]
outputdf2 = testdf.iloc[:, inputfeatures:df.shape[1]-1]


X_train, X_test, y_train, y_test = train_test_split(inputdf2, outputdf2, test_size=0.60, random_state=1)

X_train = pd.concat([X_train, inputdf1])
y_train = pd.concat([y_train, outputdf1])

stop = EarlyStopping(monitor = 'val_loss', patience = 30)
model = Sequential()
model.add(Dense(350, activation = 'sigmoid', input_dim = inputdf.shape[1]))
model.add(Dense(350, activation = 'sigmoid'))
model.add(Dense(350, activation = 'sigmoid'))
model.add(Dense(outputdf.shape[1], activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')
model.fit(X_train, y_train, epochs = 100, verbose = True, batch_size = 500, validation_data = [X_test, y_test], callbacks = [])


prediction = model.predict(X_test)
y_test = np.array(y_test)

result = []
for counter in range(0, prediction.shape[0]):
    maxval = max(prediction[counter, :])
    temp = prediction[counter,:]/maxval
    temp2 = []
    for counter2 in temp:
        if (counter2 == 1):
            temp2.append(1)
        else:
            temp2.append(0)
    result.append(temp2)

result = np.array(result)

accuracy = 0
for counter in range(0, prediction.shape[0]):
    if (list(result[counter, :]) == list(y_test[counter, :])):
        accuracy = accuracy + 1

finalacc = float(accuracy)*np.float(100)/float(prediction.shape[0])
print(finalacc)
