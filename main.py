# ==================== 3. MODEL BUILDING ====================
print("\n" + "="*50)
print("LINEAR REGRESSION MODEL")
print("="*50)

# Train Linear Regression model
print("\nTraining Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

# Make predictions
y_train_pred = lr_model.predict(X_train_scaled)
y_test_pred = lr_model.predict(X_test_scaled)

print("Model trained successfully!")
print(f"\nIntercept: {lr_model.intercept_:.2f}")
print("\nCoefficients:")
for col, coef in zip(X.columns, lr_model.coef_):
    print(f"  {col}: {coef:.4f}")