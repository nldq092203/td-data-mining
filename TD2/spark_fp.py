from pyspark.sql import SparkSession
from pyspark.ml.fpm import FPGrowth

# Correct SparkSession creation for spark-submit
spark = SparkSession.builder.appName("FP-Growth Example").getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

transactions = [
    ["Bread", "Milk"],
    ["Bread", "Diapers", "Beer", "Eggs"],
    ["Milk", "Diapers", "Beer", "Coke"],
    ["Bread", "Milk", "Diapers", "Beer"],
    ["Bread", "Milk", "Diapers", "Coke"],
]

df = spark.createDataFrame([(t,) for t in transactions], ["items"])

fp = FPGrowth(itemsCol="items", minSupport=0.6, minConfidence=0.0)
model = fp.fit(df)

model.freqItemsets.show(truncate=False)

spark.stop()
