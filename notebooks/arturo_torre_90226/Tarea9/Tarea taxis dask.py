
# coding: utf-8

# In[1]:


#Cargamos los datos en formato CSV y las librerías necesarias para hacer la tarea
from dask import dataframe
from dask import delayed #Para ejecutar en paralelo
from sklearn import linear_model #Módulo de regresión lineal
from sklearn.model_selection import train_test_split #Para dividir la muestra en train y test
import matplotlib.pyplot as plt # Para graficar
import pandas as pd
from time import time
from scipy.stats import randint as sp_randint
from scipy import stats
from distributed import Client
import distributed.joblib
from sklearn.externals import joblib
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression
import timeit
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection as ms
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


# In[2]:


trips_df = pd.read_csv("/data/trips.csv")
trips_df.head()


# In[3]:


#Corroboramos que el data frame sea formato DASK
type(trips_df)


# In[4]:


#La intención es analizar los viajes con un fare mayor que cero por lo que los siguientes chunks serán para quitarlos
trips_df.describe()


# In[5]:


#Quitamos los viajes con cero de fare_amount

trips_df = trips_df[trips_df.fare_amount > 0]


# In[6]:


#Volvemos a ver las métricas básicas y vemos que hay 5 viajes menos que corresponden a los que tienen 0 de fare
trips_df.describe()


# In[7]:


#Como lo que nos interesa es hacer un modelo para predecir la proporición de la propina en los viajes, creamos una columna con esa variable con ayuda de una función lambda
trips_df["prop"] = trips_df.apply(lambda trip: trip["tip_amount"]/ trip["fare_amount"], axis=1)


# In[8]:


#Volvemos a ver los datos para ver la nueva columna
trips_df.head()


# In[9]:


#Dividimos nuestra muestra en train y test en 80% para el train y 20% para el test para la regresión lineal NO Dask

x_train, x_test, y_train, y_test = train_test_split(
    trips_df[["trip_distance", "passenger_count", "fare_amount"]],     
    trips_df[["prop"]],  
    test_size=0.2)


# In[10]:


#Hacemos lo mismo para los datos de test
#x_test, y_test = test[["trip_distance", "passenger_count", "fare_amount"]], test["prop"]


# In[11]:


#Se crea el obejeto reg
reg = Ridge()


# In[12]:


param_grid = [{'alpha': [5, 10, 15],
                'tol': [1e-4, 1e-3, 1e-2]}]


# In[13]:


#Hacemos el grid search con cross validation para el modelo con SKlearn

sk_grid_search = GridSearchCV(reg, param_grid=param_grid, n_jobs=-1, cv=5)


# In[14]:


#Hacemos el fit con los datos correspondientes y calculamos el tiempo para después comparar
ini = timeit.default_timer()

lr_ridge = sk_grid_search.fit(x_train,y_train)

print("Tiempo de ejecución: " +  str(timeit.default_timer() - ini))


# In[15]:


#Sacamos el mejor estimador con el alpha correspondiente
lr_ridge.best_estimator_


# In[16]:


#Ahora hacemos una regresión normal para lo que se crea el obejeto reg otra vez
reg = linear_model.LinearRegression()


# In[17]:


reg.fit(x_train, y_train)


# In[18]:


#Imprimimos lo coeficientes
print('Coefficients: \n', reg.coef_)


# In[19]:


#Hacemos también los datos para Dask para que puedan ser utilizados para la regresión lineal
from dask import dataframe
trips_df = dataframe.read_csv("/data/trips.csv")
#Volvemos a quitar los ceros
trips_df = trips_df[trips_df.fare_amount > 0]

#Volvemos a crear la variable prop
trips_df["prop"] = trips_df.apply(lambda trip: trip["tip_amount"]/ trip["fare_amount"], axis=1, meta=("prop","int"))

train, test = trips_df.random_split([0.8, 0.2], random_state=2)

#Volvemos a hacer feature selection
x_train, y_train = train[["trip_distance", "passenger_count", "fare_amount"]], train["prop"] 
x_test, y_test = test[["trip_distance", "passenger_count", "fare_amount"]], test["prop"]


# In[20]:


#Ahora para hacerlo con Dask importamos lo siguiente:
from dask_ml.linear_model import LinearRegression
from dask_ml.model_selection import GridSearchCV

x_train = x_train.values.compute()
y_train = y_train.values.compute()


# In[21]:


#Ahora vamos a introducir el grid con cross 
#validation para esta parte nos vamos a basar en las funciones que aparece en manual 
#dask-ml Documentation versión 0.1 que encontré en internet (a partir de la página 12)

regdask = LinearRegression()

param_grid = [{'C': [5, 10, 15],
                'tol': [1e-4, 1e-3, 1e-2]}]


# In[22]:


dk_grid_search = GridSearchCV(regdask, param_grid=param_grid, n_jobs=-1, cv=5)


# In[23]:


#Para correr las cosas en distribuido agregamos el siguiente código:
from dask.distributed import Client
client = Client("scheduler:8786")


# In[24]:



ini = timeit.default_timer()

lr_dask = dk_grid_search.fit(x_train,y_train)

print("Tiempo de ejecución: " +  str(timeit.default_timer() - ini))


# In[25]:


lr_dask.best_estimator_


# In[26]:


#PAra el caso de Dask también corremos un regresión
regdask = LinearRegression()
regdask.fit(x_train, y_train)


# In[27]:


#Imprimimos lo coeficientes
print('Coefficients: \n', regdask.coef_)


# In[ ]:


#Vemos que los coeficientes se parecen y que tienen el mismo signo con la regresión tradicional


# In[ ]:


#Con ayuda del paquete time vemos que en Dask (distribuido) es más lento ya que toma 11.9 segundos
#comparado con los 0.5 que toma con sklearn

