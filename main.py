# ==================== 4. MODEL EVALUATION ====================
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Training metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print("\nTraining Metrics:")
print(f"  MAE: ${train_mae:.2f}")
print(f"  RMSE: ${train_rmse:.2f}")
print(f"  R² Score: {train_r2:.4f}")

# Testing metrics
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print("\nTesting Metrics:")
print(f"  MAE: ${test_mae:.2f}")
print(f"  RMSE: ${test_rmse:.2f}")
print(f"  R² Score: {test_r2:.4f}")