from dask.distributed import Client
from dask import delayed
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import numpy as np
import pandas as pd
import math

# Se crean nuevas variables de pronóstico y se procesa la base
#client = Client("scheduler:8786")
datos = pd.read_csv('Data/trips.csv')
datos['tip_prop'] = datos.tip_amount/datos.fare_amount
datos['tpep_dropoff_datetime'] = datos.tpep_dropoff_datetime\
.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
datos['tpep_pickup_datetime'] = datos.tpep_pickup_datetime\
.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
datos['car'] = 0
datos.loc[datos.car_type == 'A', 'car'] = 1
datos.loc[datos.car_type == 'B', 'car'] = 2
diff = []

# Se buscan valores nulos en la base
# Hay valores nulos en la proporción de propina

nulos = datos.loc[pd.isnull(datos.tip_prop)].index.tolist()
datos = datos.drop(nulos).reset_index(drop=True)


for i in range(len(datos)):
    aux = (datos.tpep_dropoff_datetime[i] - datos.tpep_pickup_datetime[i])
    diff.append(aux.seconds/60)

datos['time_diff'] = diff
datos['hour'] = datos.tpep_pickup_datetime.apply(lambda x: x.hour)
base = datos.drop(['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'tip_prop', 'car_type'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(base, datos.tip_prop, test_size=0.3, random_state=109649)

Regr = RandomForestRegressor(bootstrap=True, n_jobs=-1)

parameters = {'n_estimators': [3, 5, 10], 'max_depth':[4,5,6], 'min_samples_split':[2, 3, 4],
 'min_samples_leaf':[5, 6, 7], 'max_features':[6, 7, 8], 'max_leaf_nodes':[10, 11, 15, 20]}

predict_model_seq = GridSearchCV(Regr, parameters, cv=4)
predict_model_paral = GridSearchCV(Regr, parameters, cv=4)

# Búsqueda secuencial.
%time predict_model_seq.fit(X_train, y_train)

# Búsqueda en paraleleo
%time delayed(predict_model_paral.fit(X_train, y_train))