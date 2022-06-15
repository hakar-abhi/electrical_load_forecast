import glob
import numpy as np
import pandas as pd

total_data= []

for f in glob.glob ('Data-VIC/*.csv'):
    
    df = pd.read_csv(f, header = 0)
    total_data.append(df)

df = pd.concat(total_data)

df.to_csv('final_data.csv')

# print(df.dtypes)

df['Date Time'] = pd.to_datetime(df['SETTLEMENTDATE'])
df['Day'] = df['Date Time'].dt.day
df['Month'] = df['Date Time'].dt.month
df['Year'] = df['Date Time'].dt.year
df['Hour'] = df['Date Time'].dt.hour
df['Minute'] = df['Date Time'].dt.minute

df['Demand'] = pd.to_numeric(df['TOTALDEMAND'], errors = 'coerce')

df.drop(['REGION'], axis = 1, inplace = True)
df.drop(['TOTALDEMAND'], axis = 1, inplace = True)
df.drop(['SETTLEMENTDATE'], axis = 1, inplace = True)
df.drop(['RRP'], axis = 1, inplace = True)
df.drop(['PERIODTYPE'], axis = 1, inplace = True)
df.drop(['Date Time'], axis = 1, inplace = True)

print(df.head())

X = []
y = []
# print(df.shape[0])

for i in range(0, df.shape[0]-48):
    
    X.append(df.iloc[i:i+48,5])
    y.append(df.iloc[i+48,5])

X = np.array(X, dtype = np.float32)
y =  np.array(y, dtype= np.float32)

y = np.reshape(y,(len(y),1))

X = np.delete(X, list(range(1,X.shape[1],2)), axis = 1)
X = np.delete(X, list(range(1,X.shape[0],2)), axis = 0)

y = np.delete(y, list(range(1,y.shape[0],2)), axis = 0)

pd.DataFrame(X).to_csv('Revised_Data.csv')
pd.DataFrame(y).to_csv('label_data.csv')

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))

print(X.shape)

X = scaler.fit_transform(X)
y = scaler.fit_transform(y)


X_train, X_test = X[:-400], X[-400:]
y_train, y_test = y[:-400], y[-400:]

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, CuDNNLSTM
from tensorflow.keras import optimizers

model = Sequential()

model.add(CuDNNLSTM(50, return_sequences = True, input_shape = (X_train.shape[1],1)))
model.add(CuDNNLSTM(50, return_sequences = False))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))
# model.add(tf.keras.Input(shape = (X_train.shape[1],1)))

# model.add(Dense(50, activation = 'relu'))
# model.add(Dense(1))


from keras.callbacks import ModelCheckpoint, EarlyStopping
filepath ='models/{epoch:02d}-{loss:.4f}-{mae:.4f}-{val_loss:.4f}-{val_mae:.4f}.hdf5'

callbacks = [EarlyStopping(monitor='val_loss', patience = 50),
ModelCheckpoint(filepath, monitor='loss', save_best_only=True, mode = 'min')]

optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# model.fit(X_train, y_train, validation_split = 0.2, epochs = 1000, callbacks = callbacks, batch_size=8)

model.load_weights('models/bfits.hdf5')

import time

for i in range(0, X_test.shape[0]):

    demand_summary = []

    X_input = X_test[i,:,:]
    X_input = np.reshape(X_input, (1,X_input.shape[0],1))
    
    X_input = model.predict(X_input)

    forecast = scaler.inverse_transform(X_input)

    y_input = y_test[i,:]

    y_input = np.reshape(y_input, (1,1))
    actual = scaler.inverse_transform(y_input)
    
    demand_summary.append(actual)

    demand_summary.extend(forecast)
    
    df_animate = pd.DataFrame(demand_summary)

    df_animate = df_animate.T

    df_animate.to_csv('realtime_demand.csv', mode='a', header = False, index= False)

    print(demand_summary)
    
    time.sleep(0.5)



     