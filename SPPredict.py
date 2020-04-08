import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import rmse
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import warnings
import xlrd
warnings.filterwarnings("ignore")

file = pd.read_excel('C:\Datasets\SPData.xlsx', sheet_name='Sheet2')

file = file.set_index('Date')

train, test = file[:-100], file[-100:]

scaler = MinMaxScaler()
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)

n_input = 12
n_features = 1

train = train.reshape((len(train), n_features))

generator = TimeseriesGenerator(train,train,length = n_input, batch_size = 1)

model = Sequential()
model.add(LSTM(100, activation= 'relu', input_shape = (n_input, n_features)))
model.add(Dense(1))
model.compile(optimizer= 'adam', loss = 'mse')

model.fit_generator(generator, steps_per_epoch= 1, epochs = 200, verbose = 0)

pred_list = []

batch = train[-n_input:].reshape((1,n_input,n_features))

for i in range(n_input):
    pred_list.append(model.predict(batch)[0])
    batch = np.append(batch[:,1:,:], [[pred_list[i]]], axis = 1)

prediction = pd.DataFrame(scaler.inverse_transform(pred_list), index = file[n_input:].index, columns = ['Predictions'])



#prediction = pd.DataFrame(scaler.inverse_transform(pred_list), index = file[-n_input:].index, columns = ['Predictions'])


print(prediction)
plt.plot(prediction)