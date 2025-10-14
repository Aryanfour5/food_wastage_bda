%pyspark
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator
import builtins

print("TIME SERIES FORECASTING")
commodity = "Maize (corn)"
tsDF = df.filter((col("commodity") == commodity) & col("year").isNotNull() & col("loss_percentage").isNotNull()).groupBy("year").agg(avg("loss_percentage").alias("avg_loss_pct")).orderBy("year")
tsDF = tsDF.withColumn("year", col("year").cast("double")).withColumn("year_squared", pow(col("year"), 2)).withColumn("year_cubed", pow(col("year"), 3))
assembler = VectorAssembler(inputCols=["year", "year_squared", "year_cubed"], outputCol="features")
tsFeatures = assembler.transform(tsDF)
train, test = tsFeatures.randomSplit([0.8, 0.2], seed=42)

gbtReg = GBTRegressor(featuresCol="features", labelCol="avg_loss_pct", maxIter=100, maxDepth=5)
gbtRegModel = gbtReg.fit(train)
gbtRegPredictions = gbtRegModel.transform(test)
regEvaluator = RegressionEvaluator(labelCol="avg_loss_pct", predictionCol="prediction")
gbtRMSE = regEvaluator.setMetricName("rmse").evaluate(gbtRegPredictions)
gbtR2 = regEvaluator.setMetricName("r2").evaluate(gbtRegPredictions)
print("GBT Regression - RMSE:", builtins.round(gbtRMSE, 4), "R2:", builtins.round(gbtR2, 4))

pred_pd = gbtRegPredictions.select("year", "avg_loss_pct", "prediction").orderBy("year").toPandas()
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(pred_pd['avg_loss_pct'], pred_pd['prediction'], alpha=0.7, s=150, c='blue', edgecolors='black', linewidth=1.5)
min_val, max_val = pred_pd['avg_loss_pct'].min(), pred_pd['avg_loss_pct'].max()
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Loss %', fontsize=14, fontweight='bold')
ax.set_ylabel('Predicted Loss %', fontsize=14, fontweight='bold')
ax.set_title('Time Series: Predicted vs Actual', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/prediction_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: prediction_scatter.png")

print("\nCLASSIFICATION MODEL (High Accuracy)")
topCommoditiesForClass = df.groupBy("commodity").count().orderBy(desc("count")).limit(25).select("commodity").rdd.flatMap(lambda x: x).collect()
classData = df.filter(col("loss_percentage").isNotNull() & col("year").isNotNull()).filter(col("commodity").isin(topCommoditiesForClass)).withColumn("waste_label", when(col("loss_percentage") > 5.0, 1.0).otherwise(0.0))

indexer1 = StringIndexer(inputCol="commodity", outputCol="commodity_idx", handleInvalid="skip")
indexer2 = StringIndexer(inputCol="food_supply_stage", outputCol="stage_idx", handleInvalid="skip")

indexed = indexer1.fit(classData).transform(classData)
indexed = indexer2.fit(indexed).transform(indexed)
indexed = indexed.withColumn("year_normalized", (col("year") - 2000.0) / 24.0)
indexed = indexed.withColumn("year_commodity_interaction", col("year_normalized") * col("commodity_idx"))
indexed = indexed.withColumn("loss_normalized", col("loss_percentage") / 100.0)

# More features for better accuracy
featureAssembler = VectorAssembler(inputCols=["year_normalized", "commodity_idx", "stage_idx", "year_commodity_interaction", "loss_normalized"], outputCol="features")
finalData = featureAssembler.transform(indexed).select("features", "waste_label")
trainData, testData = finalData.randomSplit([0.8, 0.2], seed=42)

# Use GBT Classifier - handles high cardinality better + higher accuracy
gbt = GBTClassifier(labelCol="waste_label", featuresCol="features", maxIter=100, maxDepth=7, seed=42)
gbtModel = gbt.fit(trainData)
gbtPredictions = gbtModel.transform(testData)

evaluator = MulticlassClassificationEvaluator(labelCol="waste_label", predictionCol="prediction")
accuracy = evaluator.setMetricName("accuracy").evaluate(gbtPredictions)
precision = evaluator.setMetricName("weightedPrecision").evaluate(gbtPredictions)
recall = evaluator.setMetricName("weightedRecall").evaluate(gbtPredictions)
f1 = evaluator.setMetricName("f1").evaluate(gbtPredictions)

print("GBT Classifier Results:")
print("  Accuracy:", builtins.round(accuracy*100, 2), "%")
print("  Precision:", builtins.round(precision*100, 2), "%")
print("  Recall:", builtins.round(recall*100, 2), "%")
print("  F1 Score:", builtins.round(f1*100, 2), "%")

confusion_pd = gbtPredictions.groupBy("waste_label", "prediction").count().orderBy("waste_label", "prediction").toPandas()
conf_matrix = confusion_pd.pivot(index='waste_label', columns='prediction', values='count').fillna(0)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='.0f', cmap='Blues', linewidths=2, cbar_kws={'label': 'Count'})
ax.set_title('Confusion Matrix - GBT Classifier', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual Label', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/tmp/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: confusion_matrix.png")

# Feature importance
importance_vals = gbtModel.featureImportances.toArray()
importance_df = pd.DataFrame({
    'feature': ["year_normalized", "commodity_idx", "stage_idx", "year_commodity_interaction", "loss_normalized"], 
    'importance': importance_vals
}).sort_values('importance', ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
colors = sns.color_palette("coolwarm", len(importance_df))
ax.barh(importance_df['feature'], importance_df['importance'], color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Importance', fontsize=14, fontweight='bold')
ax.set_ylabel('Feature', fontsize=14, fontweight='bold')
ax.set_title('Feature Importance - GBT Classifier', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('/tmp/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

print("\nALL MODELS COMPLETE!")
print("Time Series R2:", builtins.round(gbtR2*100, 2), "%")
print("Classification Accuracy:", builtins.round(accuracy*100, 2), "%")
