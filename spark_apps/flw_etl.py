from pyspark.sql import SparkSession, functions as F, Window
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("FLW-ETL") \
    .master("spark://spark-master:7077") \
    .config("spark.mongodb.read.connection.uri", "mongodb://host.docker.internal:27017") \
    .config("spark.executor.memory", "2g") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:8020") \
    .getOrCreate()
df = (
    spark.read.format("mongodb")
    .option("database", "flw")
    .option("collection", "records")
    .load()
)

# If using a CSV file locally (remove if pulling from MongoDB)

# Explicitly define numeric and categorical columns based on your schema
num_cols = ["m49_code", "year", "loss_percentage", "loss_percentage_original", "loss_quantity", "sample_size"]
cat_cols = [
    "country", "region", "cpc_code", "commodity", "activity", "food_supply_stage",
    "treatment", "cause_of_loss", "method_data_collection", "reference"
]
text_cols = ["url", "notes"]  # usually ignored in ML

# Fix data types
for col in num_cols:
    df = df.withColumn(col, F.col(col).cast("double"))

for col in cat_cols:
    df = df.withColumn(col, F.when(F.col(col).isNull() | (F.trim(F.col(col)) == ""), "UNK").otherwise(F.col(col)))

# Fill numeric nulls with 0, categorical nulls already set to UNK
df = df.fillna({col: 0.0 for col in num_cols})

# Filter for valid ranges—customize as needed
df = df.filter((F.col("loss_percentage").isNotNull()) & (F.col("loss_percentage") >= 0) & (F.col("loss_percentage") <= 100))
df = df.filter(F.col("year").isNotNull())

# Remove unnecessary columns for modeling (but keep for reporting if needed)
model_cols = num_cols + cat_cols

# Rare category handling ("Other" bucket for small classes)
for col in cat_cols:
    frequent = [r[0] for r in df.groupBy(col).count().orderBy(F.desc("count")).limit(50).collect()]
    df = df.withColumn(col, F.when(F.col(col).isin(frequent), F.col(col)).otherwise(f"OTHER_{col}"))

# Rolling mean feature
entity_cols = ["country", "commodity", "food_supply_stage"]
w = Window.partitionBy(*entity_cols).orderBy("year").rowsBetween(-2, 0)
df = df.withColumn("roll3_mean_loss_pct", F.avg("loss_percentage").over(w))
df = df.withColumn("roll3_mean_loss_pct", F.when(F.col("roll3_mean_loss_pct").isNull(), 0.0).otherwise(F.col("roll3_mean_loss_pct")))

# ML pipeline
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep") for col in cat_cols]
encoders = [OneHotEncoder(inputCols=[f"{col}_idx"], outputCols=[f"{col}_oh"], handleInvalid="keep") for col in cat_cols]
feat_cols = [f"{col}_oh" for col in cat_cols] + ["year", "loss_quantity", "sample_size", "roll3_mean_loss_pct"]
assembler = VectorAssembler(inputCols=feat_cols, outputCol="features", handleInvalid="skip")

pipeline = Pipeline(stages=indexers + encoders + [assembler])
model = pipeline.fit(df)
curated = model.transform(df).select(
    "features", "year", F.col("loss_percentage").alias("label"),
    "commodity", "country", "food_supply_stage", "cause_of_loss"
).filter(F.col("features").isNotNull())

# Data preview
curated.show(10)

# Save outputs to HDFS and CSV for analysis
df.write.mode("overwrite").parquet("hdfs://namenode:8020/user/flw/cleaned_data")
curated.write.mode("overwrite").parquet("hdfs://namenode:8020/user/flw/curated_data")
df.limit(2000).toPandas().to_csv("/opt/spark-data/sample_data.csv", index=False)
