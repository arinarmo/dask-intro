

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import RFormula
import numpy as np
import time
```


```python
spark = SparkSession.builder.master("local[*]").getOrCreate()
```

Primero se lee la base guardada previamente en el notebook de dask


```python
data = spark.read.csv("0.part", header =True, inferSchema = True)

data = data.drop("_c0")
data.show()
```

    +-----------+---------------+-------------+-------------------+----------+----------+
    |fare_amount|passenger_count|trip_distance|                  y|car_type_A|car_type_B|
    +-----------+---------------+-------------+-------------------+----------+----------+
    |       22.0|              1|          6.9|0.20909090909090908|         1|         0|
    |        9.0|              1|         1.81|                0.0|         1|         0|
    |        7.5|              1|         0.96|0.13333333333333333|         1|         0|
    |        8.5|              1|          1.9|0.11764705882352941|         1|         0|
    |        7.5|              1|          1.0|0.22133333333333333|         1|         0|
    |        9.5|              5|         1.71|0.15789473684210525|         1|         0|
    |        8.0|              1|         1.27|             0.1875|         1|         0|
    |        7.5|              4|         1.55|0.21333333333333335|         1|         0|
    |        6.0|              5|         0.54|               0.26|         1|         0|
    |       52.0|              1|        15.38| 0.5769230769230769|         1|         0|
    |       28.5|              5|         7.36|0.10526315789473684|         1|         0|
    |       32.0|              1|          7.8|            0.15625|         1|         0|
    |       14.0|              1|          3.6|                0.0|         1|         0|
    |       12.5|              1|         3.37|                0.0|         1|         0|
    |       14.0|              1|         3.28|0.10714285714285714|         1|         0|
    |        5.5|              1|          0.6|                0.0|         1|         0|
    |       10.5|              3|          2.1|0.22380952380952382|         1|         0|
    |       17.5|              1|          5.7|                0.0|         0|         1|
    |       11.0|              1|          1.7|0.29090909090909095|         0|         1|
    |        6.5|              5|          1.5|                0.2|         0|         1|
    +-----------+---------------+-------------+-------------------+----------+----------+
    only showing top 20 rows
    


Se crea la formula de la regresion (para no tener que generar las variables label y features manualmente)


```python
f = RFormula(formula = "y ~ .")
data = f.fit(data).transform(data)
```

Se crea el grid


```python
ridge = LinearRegression(elasticNetParam=0.0)

paramGrid = ParamGridBuilder().addGrid(ridge.regParam,[0.01, 0.03, 0.05, 0.07, 0.09, 0.1]).build()

evaluator = RegressionEvaluator(metricName='mse')
cv = CrossValidator(estimatorParamMaps=paramGrid,
                   estimator = ridge,
                   evaluator = evaluator,
                   numFolds = 10,
                   parallelism = 4)
```

Se ejecuta el grid


```python
%%time
ridge_cv = cv.fit(data)
```

    CPU times: user 2.99 s, sys: 955 ms, total: 3.95 s
    Wall time: 20.9 s



```python
print("Loss: %f"%(np.min(ridge_cv.avgMetrics)))
```

    Loss: 0.016302

