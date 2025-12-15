from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import joblib
import os

# ===================== 1. CONFIGURATION =====================
app = Flask(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))

# File paths
MODEL_PATH = os.path.join(current_dir, "models", "model.joblib")
DATA_PATH = os.path.join(current_dir, "data", "DATASET.csv") 

# Global variables
MODEL = None
GLOBAL_FEATURES = {}
DATASET_DF = pd.DataFrame() # Holds the entire CSV data

# --- Feature Extraction and Data Loading ---
def load_data_and_extract_features():
    """Loads CSV for filtering and extracts unique features for dropdowns."""
    global GLOBAL_FEATURES, DATASET_DF
    try:
        # Load the entire dataset into memory for filtering
        df = pd.read_csv(DATA_PATH)
        
        # Normalize and keep the original column names (lowercase) for Pandas filtering
        DATASET_DF = df.copy() 
        DATASET_DF['brand'] = DATASET_DF['brand'].str.upper()
        DATASET_DF['cartype'] = DATASET_DF['cartype'].str.upper()
        DATASET_DF['fueltype'] = DATASET_DF['fueltype'].str.upper()
        DATASET_DF['status'] = DATASET_DF['status'].str.upper()

        # Extract unique values for dropdowns
        GLOBAL_FEATURES['brands'] = sorted(DATASET_DF['brand'].unique().tolist())
        GLOBAL_FEATURES['cartypes'] = sorted(DATASET_DF['cartype'].unique().tolist())
        GLOBAL_FEATURES['fueltypes'] = sorted(DATASET_DF['fueltype'].unique().tolist())
        GLOBAL_FEATURES['statuses'] = sorted(DATASET_DF['status'].unique().tolist())
        
        print(f"✅ Loaded {len(DATASET_DF)} records and extracted features.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Failed to load data from '{DATA_PATH}': {e}. Recommendation feature disabled.")
        # Fallback features if CSV loading fails
        GLOBAL_FEATURES['brands'] = ['TOYOTA', 'BMW']
        GLOBAL_FEATURES['cartypes'] = ['SUV', 'SEDAN']
        GLOBAL_FEATURES['fueltypes'] = ['PETROL', 'DIESEL']
        GLOBAL_FEATURES['statuses'] = ['NEW', 'USED']
        # DATASET_DF remains an empty DataFrame

# --- Initialization: Load Model and Features ---
try:
    MODEL = joblib.load(MODEL_PATH)
    print("🤖 Prediction Model loaded successfully!")
except Exception as e:
    print(f"❌ ERROR: Model not found or failed to load: {e}")

load_data_and_extract_features() # Run to populate data and features on startup

# ===================== 2. INPUT NORMALIZATION =====================

def normalize_inputs_for_filter(s):
    """Normalize user inputs for filtering."""
    data = {}
    # Convert all inputs to uppercase for guaranteed matching against the DataFrame
    data["BRAND"] = s.get("brand", "").upper()
    data["BUDGET_MIN"] = s.get("budget_min")
    data["BUDGET_MAX"] = s.get("budget_max")
    data["FUEL"] = s.get("fueltype", "").upper()
    data["TYPE"] = s.get("cartype", "").upper()
    data["STATUS"] = s.get("condition", "").upper() 
    
    return data

# ===================== 3. ML PREDICTION LOGIC (Unchanged) =====================

def get_prediction(data):
    """Single car price prediction with conditional guardrail."""
    if MODEL is None:
        raise Exception("Model not loaded. Please train the model first.")
        
    try:
        mileage_val = int(data["mileage"])
        year_val = int(data["year"])
        enginesize_val = float(data["enginesize"])
        status_val = data["status"]
        
        if status_val.upper() != 'NEW' and mileage_val < 1000:
            raise ValueError("Used cars must have a mileage of at least 1,000 km.")
            
        if year_val < 2000:
            raise ValueError("Model Year must be 2000 or newer for accurate prediction.")

    except ValueError as e:
        raise ValueError(f"Invalid input: {e}")

    input_data = pd.DataFrame({
        "brand": [data["brand"]], "cartype": [data["cartype"]], "fueltype": [data["fueltype"]],
        "enginesize": [enginesize_val], "status": [status_val], "year": [year_val],
        "mileage": [mileage_val]
    })
    
    predicted_price = MODEL.predict(input_data)[0]
    return int(max(1000, predicted_price))

# ===================== 4. PANDAS FILTERING LOGIC (Recommendation - CASCADING) =====================

def get_recommendations_from_csv(user_inputs):
    """Filters the loaded Pandas DataFrame based on user preferences using cascading tiers."""
    if DATASET_DF.empty:
        raise Exception("Dataset not loaded. Check DATASET.csv file path.")

    s = normalize_inputs_for_filter(user_inputs)
    
    try:
        budget_min = int(s.get("BUDGET_MIN", 0))
        budget_max = int(s.get("BUDGET_MAX", 9999999))
    except (ValueError, TypeError):
         raise ValueError("Invalid budget values submitted.")

    # Base Filter (Always apply Price filter)
    base_filter = (DATASET_DF['price'] >= budget_min) & (DATASET_DF['price'] <= budget_max)
    
    
    # --- TIER 1: STRICT MATCH (All selected filters) ---
    current_filter = base_filter
    
    # Apply all four categorical filters
    if s["BRAND"] and s["BRAND"] != "ANY":
        current_filter = current_filter & (DATASET_DF['brand'] == s["BRAND"])
    if s["FUEL"] and s["FUEL"] != "ANY":
        current_filter = current_filter & (DATASET_DF['fueltype'] == s["FUEL"])
    if s["TYPE"] and s["TYPE"] != "ANY":
        current_filter = current_filter & (DATASET_DF['cartype'] == s["TYPE"])
    if s["STATUS"] and s["STATUS"] != "ANY":
        current_filter = current_filter & (DATASET_DF['status'] == s["STATUS"])
        
    filtered_df = DATASET_DF[current_filter]
    
    # --- TIER 2: RELAXED MATCH (Relax Fuel and Status, keep Price/Brand/Type) ---
    if filtered_df.empty:
        print("Tier 1: Strict Match failed. Relaxing Fuel/Status.")
        
        # Start with base price filter
        relaxed_filter = base_filter
        
        # Re-apply Brand and Type (core requirements)
        if s["BRAND"] and s["BRAND"] != "ANY":
            relaxed_filter = relaxed_filter & (DATASET_DF['brand'] == s["BRAND"])
        if s["TYPE"] and s["TYPE"] != "ANY":
            relaxed_filter = relaxed_filter & (DATASET_DF['cartype'] == s["TYPE"])
            
        # NOTE: This will return all cars in the price range matching the core Brand/Type.
        filtered_df = DATASET_DF[relaxed_filter]
        
    # --- TIER 3: BROAD MATCH (Only Price and Brand) ---
    if filtered_df.empty:
        print("Tier 2: Relaxed Match failed. Trying broad match (Price/Brand only).")
        
        broad_filter = base_filter
        
        # Re-apply only Brand
        if s["BRAND"] and s["BRAND"] != "ANY":
            broad_filter = broad_filter & (DATASET_DF['brand'] == s["BRAND"])
            
        # NOTE: If Brand is also 'ANY', this returns ALL cars in the price range.
        filtered_df = DATASET_DF[broad_filter]


    # 2. Return Results (Final safety check for empty frame)
    if filtered_df.empty:
        return []

    # Sort by price and limit to 50 results
    recommended_cars = filtered_df.sort_values(by='price', ascending=False).head(50)
    
    # Pandas to_dict('records') returns lowercase keys by default.
    return recommended_cars.to_dict('records')

# ===================== 5. WEB ROUTES =====================

# Injects dynamic features into all templates
@app.context_processor
def inject_global_features():
    return dict(features=GLOBAL_FEATURES)

# Route 1: Home page (Entry Point)
@app.route("/", methods=["GET"])
def home_select():
    return render_template("home.html") 

# Route 2: Option 1 - Prediction Form
@app.route("/prediction", methods=["GET"])
def prediction_form():
    return render_template("prediction_form.html")

# Route 3: Option 1 - Prediction Calculation
@app.route("/predict", methods=["POST"])
def predict():
    try:
        price = get_prediction(request.form)
        return render_template(
            "results.html", 
            predicted_price=f"${price:,.2f}", 
            input_data=request.form 
        )
    except Exception as e:
        return render_template("results.html", error_message=f"Error: {e}")

# Route 4: Option 2 - Recommendation Form
@app.route("/recommendation", methods=["GET"])
def recommendation_form():
    return render_template("recommendation_form.html") 

# Route 5: Option 2 - Recommendation Calculation
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        recommendations = get_recommendations_from_csv(request.form)
        
        if not recommendations:
             return render_template("recommendations.html", error_message="No car found matching your criteria. Try broadening your budget or filters.")

        return render_template(
            "recommendations.html", 
            recommendations=recommendations,
            total_matches=len(recommendations)
        )
    except Exception as e:
        return render_template("recommendations.html", error_message=f"Recommendation Error: {e}")

# Route 6: Contact Page 
@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == 'POST':
        success_message = "Thank you for contacting us! We will respond to your inquiry shortly."
        return render_template("contact.html", success_message=success_message)
    return render_template("contact.html")

# Route 7: Explore Page 
@app.route("/explore", methods=["GET"])
def explore():
    return render_template("explore.html")


if __name__ == "__main__":
    os.makedirs(os.path.join(current_dir, "templates"), exist_ok=True) 
    # NOTE: Ensure you have a 'data' folder next to app.py containing DATASET.csv
    os.makedirs(os.path.join(current_dir, "data"), exist_ok=True) 
    print("Server starting at http://127.0.0.1:5000")
    app.run(debug=True)