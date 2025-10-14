from pyspark.sql import SparkSession, functions as F, Window

spark = SparkSession.builder.appName("FLW-train-test-split").getOrCreate()
df = spark.read.parquet("hdfs:///flw/cleaned_preprocessed")
entity = ["country","commodity","food_supply_stage"]
w = Window.partitionBy(*entity)
df = df.withColumn("max_year", F.max("year").over(w))

train = df.filter(F.col("year") < F.col("max_year"))
test = df.filter(F.col("year") == F.col("max_year"))

train.write.mode("overwrite").parquet("hdfs:///flw/train_data")
test.write.mode("overwrite").parquet("hdfs:///flw/test_data")
print("Created train/test forecast splits.")
