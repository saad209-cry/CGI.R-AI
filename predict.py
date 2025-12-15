import pandas as pd
import joblib
import os
import sys

# ---------------- CONFIG ----------------
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "models", "model.joblib")

# ---------------- PREDICTION FUNCTION ----------------
def predict_price(brand, cartype, fueltype, enginesize, status, year, mileage):
    """
    Loads the trained model and predicts the car price based on 7 features.
    """
    # 1. Load the Model
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return "ERROR: Model file not found. Run trainlinear.py first.", None

    # 2. Prepare the Input Data (MUST match training data format)
    input_data = pd.DataFrame({
        "brand": [brand],
        "cartype": [cartype],
        "fueltype": [fueltype],
        "enginesize": [enginesize],
        "status": [status],
        "year": [year],
        "mileage": [mileage]
    })
    
    # 3. Predict the Price
    try:
        predicted_price = model.predict(input_data)[0]
        # Format the price to an integer
        return None, int(predicted_price)
    except Exception as e:
        return f"ERROR during prediction: {e}", None

# ---------------- STANDALONE TEST ----------------
if __name__ == "__main__":
    print("--- PREDICTION SCRIPT TEST ---")
    
    # Example Car Input
    # This is the test case that should be run from your website!
    test_car = {
        "brand": "BMW",
        "cartype": "SUV",
        "fueltype": "Hybrid",
        "enginesize": 3.0,
        "status": "Used",
        "year": 2022,
        "mileage": 15000
    }
    
    print(f"Input Car: {test_car['year']} {test_car['brand']} {test_car['cartype']}")
    
    # Make the prediction
    error, price = predict_price(**test_car)
    
    if error:
        print(f"Result: {error}")
    else:
        print(f"✅ Predicted Price: ${price:,.2f}")
        print("This is the value your website will display.")

    # Second Test: A cheap, old car
    test_car_old = {
        "brand": "Honda",
        "cartype": "Sedan",
        "fueltype": "Petrol",
        "enginesize": 1.5,
        "status": "Used",
        "year": 2017,
        "mileage": 89000
    }
    
    print(f"\nInput Car: {test_car_old['year']} {test_car_old['brand']} {test_car_old['cartype']}")
    error, price_old = predict_price(**test_car_old)
    
    if error:
        print(f"Result: {error}")
    else:
        print(f"✅ Predicted Price: ${price_old:,.2f}")