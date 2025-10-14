from pyspark.sql import SparkSession, functions as F, Window
spark = SparkSession.builder.getOrCreate()
entity = ["country","commodity","food_supply_stage"]
df = spark.read.parquet("hdfs:///flw/raw_with_cats")
w = Window.partitionBy(*entity)
df = df.withColumn("max_year", F.max("year").over(w))
train = df.filter(F.col("year") < F.col("max_year"))
test = df.filter(F.col("year") == F.col("max_year"))

train.write.mode("overwrite").parquet("hdfs:///flw/forecast_train")
test.write.mode("overwrite").parquet("hdfs:///flw/forecast_test")
