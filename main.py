import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# ==================== 1. LOAD DATA ====================
print("Loading data...")
df = pd.read_csv('insurance_data_linear.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ==================== 2. DATA PREPROCESSING ====================
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Create a copy for preprocessing
df_processed = df.copy()

# Encode categorical variables
print("\n[Step 1] Encoding categorical variables...")
categorical_cols = ['sex', 'smoker', 'region']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    print(f"  {col} encoded")

# Separate features and target
print("\n[Step 2] Separating features and target...")
X = df_processed.drop('charges', axis=1)
y = df_processed['charges']

print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")

# Split data into training and testing sets
print("\n[Step 3] Splitting data (70/30)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Testing samples: {X_test.shape[0]}")

# Scale features
print("\n[Step 4] Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  Features scaled!")

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