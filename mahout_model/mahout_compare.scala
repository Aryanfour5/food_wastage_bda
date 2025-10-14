%pyspark
import builtins
import numpy as np

print("\n=== MODEL COMPARISON & TESTING ===")

# ========== 1. MODEL COMPARISON CHART ==========
print("\n1. Creating Model Comparison Chart...")
comparison_data = pd.DataFrame({
    'Model': ['Time Series\n(GBT Regressor)', 'Classification\n(GBT Classifier)'],
    'Accuracy/R2': [gbtR2 * 100, accuracy * 100],
    'Type': ['Regression', 'Classification']
})

fig, ax = plt.subplots(figsize=(12, 7))
colors = ['#3498db', '#e74c3c']
bars = ax.bar(comparison_data['Model'], comparison_data['Accuracy/R2'], color=colors, edgecolor='black', linewidth=2, width=0.6)
ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3, linestyle='--')

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{comparison_data.iloc[i]["Accuracy/R2"]:.2f}%',
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2., height/2,
            comparison_data.iloc[i]['Type'],
            ha='center', va='center', fontsize=11, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('/tmp/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: model_comparison.png")

# ========== 2. REGRESSION - SAMPLE PREDICTIONS ==========
print("\n2. Testing Time Series Model on Sample Data...")
print("\n--- REGRESSION MODEL PREDICTIONS ---")
print("Predicting food loss for Maize (corn) by year:\n")

sample_predictions = gbtRegPredictions.select("year", "avg_loss_pct", "prediction").orderBy("year").toPandas()
sample_predictions['error'] = np.abs(sample_predictions['avg_loss_pct'] - sample_predictions['prediction'])
sample_predictions['accuracy_pct'] = 100 - (sample_predictions['error'] / sample_predictions['avg_loss_pct'] * 100)

print("Sample Predictions (First 10 years):")
print("-" * 80)
print(f"{'Year':<8} {'Actual Loss %':<15} {'Predicted %':<15} {'Error':<12} {'Accuracy':<12}")
print("-" * 80)
for idx, row in sample_predictions.head(10).iterrows():
    print(f"{int(row['year']):<8} {row['avg_loss_pct']:>12.2f}% {row['prediction']:>12.2f}% {row['error']:>10.2f}% {row['accuracy_pct']:>10.1f}%")
print("-" * 80)
print(f"Average Prediction Accuracy: {sample_predictions['accuracy_pct'].mean():.2f}%\n")

# Residual Plot
fig, ax = plt.subplots(figsize=(12, 7))
residuals = sample_predictions['avg_loss_pct'] - sample_predictions['prediction']
ax.scatter(sample_predictions['prediction'], residuals, alpha=0.6, s=100, c='purple', edgecolors='black', linewidth=1)
ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error Line')
ax.set_xlabel('Predicted Loss %', fontsize=14, fontweight='bold')
ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=14, fontweight='bold')
ax.set_title('Regression Model - Residual Plot', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/tmp/residual_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: residual_plot.png")

# ========== 3. CLASSIFICATION - SAMPLE PREDICTIONS ==========
print("\n3. Testing Classification Model on Sample Data...")
print("\n--- CLASSIFICATION MODEL PREDICTIONS ---")
print("Predicting whether food will be wasted (loss > 5%):\n")

# Get sample predictions - convert probability to list first
sample_class_pred = gbtPredictions.select("waste_label", "prediction", "probability").limit(20).toPandas()
sample_class_pred['actual'] = sample_class_pred['waste_label'].apply(lambda x: 'High Waste' if x == 1.0 else 'Low Waste')
sample_class_pred['predicted'] = sample_class_pred['prediction'].apply(lambda x: 'High Waste' if x == 1.0 else 'Low Waste')
sample_class_pred['correct'] = sample_class_pred['waste_label'] == sample_class_pred['prediction']
# FIX: Convert DenseVector to array, then get max
sample_class_pred['confidence'] = sample_class_pred['probability'].apply(lambda x: float(np.max(x.toArray())) * 100)

print("Sample Predictions (First 20 records):")
print("-" * 95)
print(f"{'#':<4} {'Actual':<12} {'Predicted':<12} {'Confidence':<12} {'Result':<10}")
print("-" * 95)
for idx, row in sample_class_pred.iterrows():
    result = "Correct" if row['correct'] else "Wrong"
    print(f"{idx+1:<4} {row['actual']:<12} {row['predicted']:<12} {row['confidence']:>10.1f}% {result:<10}")
print("-" * 95)
correct_count = sample_class_pred['correct'].sum()
print(f"Sample Accuracy: {correct_count}/{len(sample_class_pred)} = {(correct_count/len(sample_class_pred)*100):.1f}%\n")

# ========== 4. PREDICTION DISTRIBUTION CHART ==========
print("\n4. Creating Prediction Distribution Chart...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Actual distribution
actual_counts = gbtPredictions.groupBy("waste_label").count().toPandas()
colors1 = ['#2ecc71', '#e74c3c']
ax1.pie(actual_counts['count'], labels=['Low Waste', 'High Waste'], autopct='%1.1f%%', 
        startangle=90, colors=colors1, explode=[0.05, 0.05], shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'})
ax1.set_title('Actual Distribution', fontsize=14, fontweight='bold')

# Predicted distribution
pred_counts = gbtPredictions.groupBy("prediction").count().toPandas()
ax2.pie(pred_counts['count'], labels=['Low Waste', 'High Waste'], autopct='%1.1f%%',
        startangle=90, colors=colors1, explode=[0.05, 0.05], shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'})
ax2.set_title('Predicted Distribution', fontsize=14, fontweight='bold')

plt.suptitle('Classification Results - Actual vs Predicted Distribution', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/tmp/prediction_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: prediction_distribution.png")

# ========== 5. FINAL SUMMARY ==========
print("\n" + "="*80)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*80)
print(f"\n1. TIME SERIES FORECASTING (Maize corn)")
print(f"   - Model: Gradient Boosted Trees Regressor")
print(f"   - R2 Score: {builtins.round(gbtR2*100, 2)}%")
print(f"   - RMSE: {builtins.round(gbtRMSE, 4)}")
print(f"   - Average Prediction Error: {builtins.round(sample_predictions['error'].mean(), 3)}%")

print(f"\n2. CLASSIFICATION (High vs Low Waste)")
print(f"   - Model: Gradient Boosted Trees Classifier")
print(f"   - Accuracy: {builtins.round(accuracy*100, 2)}%")
print(f"   - Precision: {builtins.round(precision*100, 2)}%")
print(f"   - Recall: {builtins.round(recall*100, 2)}%")
print(f"   - F1 Score: {builtins.round(f1*100, 2)}%")

print("\n" + "="*80)
print("ALL CHARTS CREATED:")
print("  1. model_comparison.png - Compare both models")
print("  2. residual_plot.png - Regression error analysis")
print("  3. prediction_distribution.png - Classification distribution")
print("  4. prediction_scatter.png - Time series predictions")
print("  5. confusion_matrix.png - Classification accuracy")
print("  6. feature_importance.png - Important features")
print("="*80)
