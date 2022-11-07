# Descripcion: Este programa utiliza una red neuronal artificial llamda memoria a corto plazo
# Para predecir el precio mas cercano de la una accion de una compañia al momento de cierre. 
#usando datos de los ultimos 60 de la empresa (Apple.Inc)

#Import the labraries
import math
import pandas_datareader as web 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Consiguiendo los datos de las acciones
df = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
#Mostrando los datos 
df

#observamos el numero de filas y columnas dentro del dataset
df.shape

#Visualizando el precio de cierre de las acciones
plt.figure(figsize=(16,8))
plt.title('Precio de cierre historico')
plt.plot(df['Close'])
plt.xlabel('Fecha', fontsize=18)
plt.ylabel('Precio de cierre USD($)', fontsize=18)
plt.show()

#Creamos un nuevo dataframe solo con la columna de precio de cierre
data = df.filter(['Close'])
#Convertimos el dataframe a un formato numpy
dataset = data.values
#Conseguimos el numero de filas para entrenar al modelo de Machine Learning 
training_data_len = math.ceil(len(dataset) * .8)

training_data_len

#Escalamos los datos
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

#Creamos el modelo de entrenamiento
#Creamos el modelo de entrenamiento escalado
train_data = scaled_data[0:training_data_len , :]
#Separamos los datos en dos ejes x_train y y_train data sets 
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<= 61:
    print(x_train)
    print(y_train)
    print()

#Convirtiendo the x_train y y_train a una serie numpy
x_train, y_train = np.array(x_train), np.array(y_train)

#Cambiamos la forma de los datos
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape

from numpy.core.fromnumeric import shape
from keras.engine.input_layer import Input
#Construyendo el Modelo LSTM 
model = Sequential()
model.add(LSTM(50, return_sequences= True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))

#Compilando el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

#Entrenando el modelo
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Creando el evaluador del dataset 
#Creando una nueva serie que contenga valores escalados del index 1543 a 2003
test_data = scaled_data[training_data_len - 60: , :]
#Creando los date sets x_test y y_test
x_test = []
y_test = dataset[training_data_len: , :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])

#Convirtiendo los datos en una matriz numpy
x_test = np.array(x_test)

#Reformando los datos
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#Consiguiendo los modelos predictivos para los valores de precios
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#Consiguiendo el margen de error de nuestro modelo con RMSE
rmse = np.sqrt( np.mean(predictions - y_test )**2)
rmse

#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualizando los datos
plt.figure(figsize=(16, 8))
plt.title('Modelo')
plt.xlabel('Fecha', fontsize=18)
plt.ylabel('Precio de cierre USD$', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['entrenamiento', 'validacion', 'prediccion'], loc='lower right')
plt.show()

#mostrando la validacion y el precio predictivo
valid

#Mirando la prediccion en un periodo de tiempo dado
apple_quote = web.DataReader('AAPL', data_source='yahoo', start='2012-01-01', end='2019-12-17')
#Creamos un nuevo dataframe
new_df = apple_quote.filter(['Close'])
#Consiguiendo los precios de cierre de los ultimos 60 dias y convirtiendo los dataframe en una matriz
last_60_days = new_df[-60:].values
#Escalamos los datos para que sean valores entre 0 y 1
last_60_days_scaled = scaler.transform(last_60_days)
#Creamos una lista vacia
X_test = []
#Apendisamos los ultimos 60 dias 
X_test.append(last_60_days_scaled)
#Convertimos X_test en una matriz numpy
X_test = np.array(X_test)
#Reformamos los datos 
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#Obtenemos el precio predictivo escalado
pred_price = model.predict(X_test)
#Deshaciendo el escalar
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

#Haciendo la cotizacion
apple_quote2 = web.DataReader('AAPL', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print(apple_quote2['Close'])


