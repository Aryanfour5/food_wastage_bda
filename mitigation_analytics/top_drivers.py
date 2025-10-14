from pyspark.sql import SparkSession, functions as F

spark = SparkSession.builder.appName("FLW-drivers").getOrCreate()
df = spark.read.parquet("hdfs:///flw/cleaned_preprocessed")
results = (df.groupBy("commodity","food_supply_stage","cause_of_loss")
    .agg(F.count("*").alias("n"),
         F.avg("loss_percentage").alias("avg_loss_pct"),
         F.expr("percentile_approx(loss_percentage, 0.9)").alias("p90_loss_pct"))
    .withColumn("priority_score", 0.5*F.col("avg_loss_pct")+0.5*F.col("p90_loss_pct"))
    .orderBy(F.desc("priority_score")))
results.write.mode("overwrite").csv("hdfs:///flw/top_loss_drivers")
print("Analytics for top loss causes exported.")
