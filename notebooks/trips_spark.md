
### Sección Spark
#### Daniel Sharp 138176
Ejecutamos el grid search en spark


```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import numpy as np
spark = SparkSession.builder.master("local[*]").getOrCreate()
```

Carga de los datos como fueron procesados en el notebook de Dask


```python
trips_df = spark.read.csv('/home/jovyan/work/0.part', header =True, inferSchema=True)
```


```python
trips_df.limit(10).toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>_c0</th>
      <th>fare_amount</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>dow_0</th>
      <th>dow_1</th>
      <th>dow_3</th>
      <th>dow_4</th>
      <th>dow_6</th>
      <th>dow_2</th>
      <th>hour_buck_4</th>
      <th>hour_buck_3</th>
      <th>hour_buck_2</th>
      <th>car_type_B</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>6.90</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.209091</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>9.0</td>
      <td>1</td>
      <td>1.81</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>7.5</td>
      <td>1</td>
      <td>0.96</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>8.5</td>
      <td>1</td>
      <td>1.90</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.117647</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>7.5</td>
      <td>1</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.221333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>9.5</td>
      <td>5</td>
      <td>1.71</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.157895</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>8.0</td>
      <td>1</td>
      <td>1.27</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7.5</td>
      <td>4</td>
      <td>1.55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.213333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>6.0</td>
      <td>5</td>
      <td>0.54</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.260000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>52.0</td>
      <td>1</td>
      <td>15.38</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.576923</td>
    </tr>
  </tbody>
</table>
</div>




```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import RFormula
```


```python
trips_df = trips_df.drop('_c0')
```

Se crea la columna label y features como las requiere Spark en sus modelos.


```python
formula = RFormula(formula = "target ~ .")
```


```python
df = formula.fit(trips_df).transform(trips_df)
```


```python
df.limit(10).toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fare_amount</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>dow_0</th>
      <th>dow_1</th>
      <th>dow_3</th>
      <th>dow_4</th>
      <th>dow_6</th>
      <th>dow_2</th>
      <th>hour_buck_4</th>
      <th>hour_buck_3</th>
      <th>hour_buck_2</th>
      <th>car_type_B</th>
      <th>target</th>
      <th>features</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>1</td>
      <td>6.90</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.209091</td>
      <td>(22.0, 1.0, 6.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,...</td>
      <td>0.209091</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>1</td>
      <td>1.81</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.000000</td>
      <td>(9.0, 1.0, 1.81, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,...</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.5</td>
      <td>1</td>
      <td>0.96</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.133333</td>
      <td>(7.5, 1.0, 0.96, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,...</td>
      <td>0.133333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>1</td>
      <td>1.90</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.117647</td>
      <td>(8.5, 1.0, 1.9, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ...</td>
      <td>0.117647</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.5</td>
      <td>1</td>
      <td>1.00</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.221333</td>
      <td>(7.5, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, ...</td>
      <td>0.221333</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.5</td>
      <td>5</td>
      <td>1.71</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.157895</td>
      <td>(9.5, 5.0, 1.71, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,...</td>
      <td>0.157895</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.0</td>
      <td>1</td>
      <td>1.27</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0.187500</td>
      <td>(8.0, 1.0, 1.27, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,...</td>
      <td>0.187500</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.5</td>
      <td>4</td>
      <td>1.55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.213333</td>
      <td>(7.5, 4.0, 1.55, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,...</td>
      <td>0.213333</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6.0</td>
      <td>5</td>
      <td>0.54</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.260000</td>
      <td>(6.0, 5.0, 0.54, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,...</td>
      <td>0.260000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>52.0</td>
      <td>1</td>
      <td>15.38</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.576923</td>
      <td>(52.0, 1.0, 15.38, 0.0, 1.0, 0.0, 0.0, 0.0, 0....</td>
      <td>0.576923</td>
    </tr>
  </tbody>
</table>
</div>



Se define el etimador y el grid de parametros que se utilizará:


```python
lr = LinearRegression(elasticNetParam=1.0)
paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [50, 100, 200]) \
    .addGrid(lr.regParam, [0.01, 0.001, 0.1,1.0])\
    .build()
evaluator = RegressionEvaluator(metricName='mse')
```

Se define el grid search con cross-validation


```python
crossval = CrossValidator(estimatorParamMaps=paramGrid,
                          estimator=lr,
                          evaluator=evaluator,
                          numFolds=10,
                         parallelism = 4)
```


```python
%%time
cvModel = crossval.fit(df)
```

    CPU times: user 5.32 s, sys: 2.4 s, total: 7.73 s
    Wall time: 21.4 s



```python
cvModel.getEstimatorParamMaps()[ np.argmin(cvModel.avgMetrics) ]
```




    {Param(parent='LinearRegression_44d9ba2666b3b763e46f', name='maxIter', doc='max number of iterations (>= 0).'): 50,
     Param(parent='LinearRegression_44d9ba2666b3b763e46f', name='regParam', doc='regularization parameter (>= 0).'): 0.001}




```python
np.min(cvModel.avgMetrics)
```




    0.016312933805217481


