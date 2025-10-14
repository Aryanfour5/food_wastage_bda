from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# SparkSession with HDFS and MongoDB configuration
spark = (
    SparkSession.builder
    .appName("FLW-ETL")
    .master("spark://spark-master:7077")  # Connect to Spark master
    .config("spark.mongodb.read.connection.uri", "mongodb://host.docker.internal:27017")
    .config("spark.executor.memory", "2g")
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020")  # HDFS configuration
    .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:10.2.0")
    .getOrCreate()
)

# Load from MongoDB
df = (
    spark.read.format("mongodb")
    .option("database", "flw")
    .option("collection", "records")
    .load()
)

print("Data loaded from MongoDB:", df.count(), "rows")

# Your existing data cleaning code...
num_cols = ["m49_code", "year", "loss_percentage", "loss_quantity", "sample_size"]
for col in num_cols:
    df = df.withColumn(col, F.col(col).cast("double"))

cat_cols = ["country", "region", "cpc_code", "commodity", "food_supply_stage",
            "activity", "treatment", "cause_of_loss", "method_data_collection", "reference"]

for col in cat_cols:
    df = df.withColumn(col, F.when(F.col(col).isNull() | (F.trim(F.col(col)) == ""), "UNK").otherwise(F.col(col)))

df = df.filter((F.col("loss_percentage").isNotNull()) & (F.col("loss_percentage") >= 0) & (F.col("loss_percentage") <= 100))
df = df.filter(F.col("year").isNotNull())

print("Rows after cleaning:", df.count())

# Save processed data to HDFS
df.write.mode("overwrite").parquet("hdfs://namenode:8020/user/flw/cleaned_data")
print("Data saved to HDFS at: hdfs://namenode:8020/user/flw/cleaned_data")

# Feature engineering...
for col in cat_cols:
    values = [r[0] for r in df.groupBy(col).count().orderBy(F.desc("count")).limit(50).collect()]
    df = df.withColumn(col, F.when(F.col(col).isin(values), F.col(col)).otherwise(F.lit(f"OTHER_{col}")))

entity_cols = ["country", "commodity", "food_supply_stage"]
w = Window.partitionBy(*entity_cols).orderBy("year").rowsBetween(-2, 0)
df = df.withColumn("roll3_mean_loss_pct", F.avg("loss_percentage").over(w))

# ML pipeline...
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in cat_cols]
encoders = [OneHotEncoder(inputCols=[f"{col}_idx"], outputCols=[f"{col}_oh"], handleInvalid="keep") for col in cat_cols]
feat_cols = [f"{col}_oh" for col in cat_cols] + ["year", "loss_quantity", "sample_size", "roll3_mean_loss_pct"]
assembler = VectorAssembler(inputCols=feat_cols, outputCol="features")
pipeline = Pipeline(stages=indexers + encoders + [assembler])
model = pipeline.fit(df)
curated = model.transform(df).select("features", "year", F.col("loss_percentage").alias("label"), "commodity", "country", "food_supply_stage", "cause_of_loss")

# Save curated data to HDFS
curated.write.mode("overwrite").parquet("hdfs://namenode:8020/user/flw/curated_data")
print("Curated data saved to HDFS")

# Export sample for analysis
sample_pd = df.limit(2000).toPandas()
print(sample_pd.head())

spark.stop()
