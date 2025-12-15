import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor  # <--- The Upgrade!
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

# ---------------- CONFIG ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(current_dir, "data", "DATASET.csv") 
MODEL_OUT = os.path.join(current_dir, "models", "model.joblib")

# ---------------- LOAD DATA ----------------
print(f"📂 Looking for data at: {DATA_FILE}")

try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print("❌ Error: Could not find DATASET.csv. Check if it is inside the 'data' folder.")
    exit()

# Ensure correct column names
df.columns = df.columns.str.lower()
df = df[["brand", "cartype", "enginesize", "fueltype", "price", "status"]]

# Convert numeric
df["enginesize"] = pd.to_numeric(df["enginesize"], errors="coerce")
df["price"] = pd.to_numeric(df["price"], errors="coerce")

# Drop rows with missing values
df.dropna(inplace=True)

# ---------------- CLEANING (THE FIX) ----------------
# We set hard limits to focus on the main market
MIN_PRICE = 1000        # $1k minimum
MAX_PRICE = 500000      # $500k maximum (Your requested cap)

print(f"🧹 Filtering Data: Keeping prices between ${MIN_PRICE:,.0f} and ${MAX_PRICE:,.0f}")

# Keep only the rows inside that range
df = df[(df["price"] > MIN_PRICE) & (df["price"] < MAX_PRICE)]

print(f"✅ Dataset ready: {len(df)} rows (Cleaned)")

# ---------------- FEATURES ----------------
X = df.drop("price", axis=1)
y = df["price"]

categorical_features = ["brand", "cartype", "fueltype", "status"]
numerical_features = ["enginesize"]

# ---------------- PIPELINE ----------------
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numerical_features)
])

# Using RANDOM FOREST for better accuracy
model = Pipeline([
    ("prep", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# ---------------- TRAIN ----------------
print("🤖 Training the Random Forest model... (This might take a moment)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATE ----------------
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nModel Performance")
print(f"MAE: ${mae:,.2f}")
print(f"R²: {r2:.4f}")

# ---------------- SAVE ----------------
os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(model, MODEL_OUT)
print(f"\n💾 Model saved to: {MODEL_OUT}")

# ---------------- USER INPUT PREDICTION TEST ----------------
sample_car = pd.DataFrame({
    "brand": ["Toyota"],
    "cartype": ["SUV"],
    "fueltype": ["Petrol"],
    "enginesize": [3.0],
    "status": ["used"]
})

predicted_price = model.predict(sample_car)[0]
print(f"\nPredicted Price for Test Car: ${predicted_price:,.2f}")