%pyspark
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.sql.functions import *

sns.set_style("whitegrid")
sns.set_palette("husl")

df = spark.read.option("header", "true").option("inferSchema", "true").csv("file:///tmp/flw_export_1.csv")
print("Total records:", df.count())

print("\nSUMMARY STATISTICS")
df.describe("loss_percentage", "year").show()

lossByCommodity_pd = df.groupBy("commodity").agg(avg("loss_percentage").alias("avg_loss"), count("*").alias("count")).filter(col("count") > 10).orderBy(desc("avg_loss")).limit(10).toPandas()
yearlyTrend_pd = df.filter(col("year").isNotNull() & col("loss_percentage").isNotNull()).groupBy("year").agg(avg("loss_percentage").alias("avg_loss")).orderBy("year").toPandas()
top_commodities = df.groupBy("commodity").count().orderBy(desc("count")).limit(5).select("commodity").rdd.flatMap(lambda x: x).collect()
multiCommodityTrend_pd = df.filter(col("commodity").isin(top_commodities) & col("year").isNotNull() & col("loss_percentage").isNotNull()).groupBy("commodity", "year").agg(avg("loss_percentage").alias("avg_loss")).orderBy("year").toPandas()

print("\nData prepared for visualization!")
%pyspark
print("Creating visualizations...")

fig, ax = plt.subplots(figsize=(14, 8))
colors = sns.color_palette("Spectral", len(lossByCommodity_pd))
ax.barh(lossByCommodity_pd['commodity'], lossByCommodity_pd['avg_loss'], color=colors, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Average Loss %', fontsize=14, fontweight='bold')
ax.set_ylabel('Commodity', fontsize=14, fontweight='bold')
ax.set_title('Top 10 Commodities by Loss', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/commodity_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: commodity_loss.png")

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(yearlyTrend_pd['year'], yearlyTrend_pd['avg_loss'], marker='o', linewidth=3, markersize=8, color='#2E86C1', markerfacecolor='#E74C3C')
ax.fill_between(yearlyTrend_pd['year'], yearlyTrend_pd['avg_loss'], alpha=0.3, color='#85C1E9')
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Loss %', fontsize=14, fontweight='bold')
ax.set_title('Food Loss Trend (2000-2024)', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('/tmp/yearly_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: yearly_trend.png")

fig, ax = plt.subplots(figsize=(16, 8))
for commodity in top_commodities:
    data = multiCommodityTrend_pd[multiCommodityTrend_pd['commodity'] == commodity]
    ax.plot(data['year'], data['avg_loss'], marker='o', linewidth=2.5, markersize=7, label=commodity)
ax.set_xlabel('Year', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Loss %', fontsize=14, fontweight='bold')
ax.set_title('Top 5 Commodities Trend', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12, frameon=True, shadow=True)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/multi_commodity_trend.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: multi_commodity_trend.png")

print("All visualizations created!")
