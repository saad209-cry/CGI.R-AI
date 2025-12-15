import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import sys

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, "data", "DATASET.csv")
MODEL_DIR = os.path.join(current_dir, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

# Ensure the models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Feature Definition ---
# These are the FINAL 7 features used for training
NUMERICAL_FEATURES = ['enginesize', 'year', 'mileage']
CATEGORICAL_FEATURES = ['brand', 'cartype', 'fueltype', 'status']
TARGET_FEATURE = 'price'

def train_and_save_model():
    """
    Loads data, preprocesses, trains the Linear Regression model,
    and saves the pipeline to model.joblib.
    """
    print("--- Starting Model Training ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {len(df)} records from DATASET.csv")
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {DATA_PATH}. Please run feeder.py first.")
        sys.exit(1)
    
    # 2. Data Cleaning and Preparation
    # Drop rows where any of the necessary columns might be missing
    features_to_check = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET_FEATURE]
    df.dropna(subset=features_to_check, inplace=True)
    
    # Ensure numerical types are correct (especially after the feeder runs)
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=NUMERICAL_FEATURES, inplace=True)
    
    print(f"Cleaned dataset size: {len(df)} records")

    # Define X and y
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_FEATURE]
    
    # 3. Define Preprocessing Steps (The ColumnTransformer handles feature conversion)
    
    # Preprocessor for numerical features (no transformation needed)
    numerical_transformer = 'passthrough' 
    
    # Preprocessor for categorical features (One-Hot Encoding)
    # This is the key step for converting text like 'BMW' into numerical features.
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop' # Drops any columns not explicitly listed above
    )
    
    # 4. Create the Full Pipeline
    # The pipeline ensures that the exact same transformations (OneHotEncoder) 
    # used during training are applied during prediction (in predict.py).
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # 5. Split Data and Train Model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print("Training model...")
    model_pipeline.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Model Training Complete.")
    print(f"R-squared Score (Accuracy): {r2:.4f}")
    print("-" * 30)
    
    # 7. Save the Final Pipeline (Model)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f"✅ Trained model saved successfully to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()