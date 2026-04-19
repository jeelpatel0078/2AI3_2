import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==================== 5. MAKE PREDICTIONS ====================
print("\n" + "="*50)
print("MAKING PREDICTIONS")
print("="*50)

# Example prediction
print("\nExample prediction for a new customer:")
new_data = pd.DataFrame({
    'age': [25],
    'sex': ['male'],
    'bmi': [28.5],
    'children': [1],
    'smoker': ['no'],
    'region': ['northeast']
})

print(f"\nInput data:\n{new_data}")

# Preprocess new data
new_data_processed = new_data.copy()
for col in categorical_cols:
    if col in new_data_processed.columns:
        new_data_processed[col] = label_encoders[col].transform(new_data_processed[col])

# Scale and predict
new_data_scaled = scaler.transform(new_data_processed)
prediction = lr_model.predict(new_data_scaled)[0]

print(f"\nPredicted Insurance Charges: ${prediction:.2f}")

print("\n" + "="*50)
print("Model Complete!")
print("="*50)