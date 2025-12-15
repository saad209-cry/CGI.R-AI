import pandas as pd
import time
import os

# ---------------- CONFIG ----------------
OUTPUT_FILE = "data/DATASET.csv"
SOURCE_FILE = "data/source_data.csv" 
DELAY_SECONDS = 0.05  # FAST SPEED: Adds ~20 cars per second
TARGET_COUNT = 15000  # Stop after adding this many cars

# ---------------- REAL DATA (STARTER PACK) ----------------
# Actual market values (Brand, Type, Fuel, Engine, Status, Year, Mileage, Price)
REAL_CARS = [
    ["Toyota", "Sedan", "Petrol", 2.5, "Used", 2021, 45000, 24500],
    ["Ford", "SUV", "Petrol", 3.5, "Used", 2019, 62000, 28900],
    ["BMW", "Sedan", "Diesel", 3.0, "Used", 2018, 58000, 31200],
    ["Audi", "SUV", "Hybrid", 2.0, "New", 2024, 150, 52000],
    ["Mercedes", "Coupe", "Petrol", 4.0, "Used", 2020, 34000, 68500],
    ["Honda", "Sedan", "Petrol", 1.5, "Used", 2017, 89000, 16500],
    ["Tesla", "Sedan", "Electric", 0.0, "Used", 2022, 12000, 41000],
    ["Hyundai", "Hatchback", "Petrol", 1.6, "Used", 2016, 95000, 11200],
    ["Kia", "SUV", "Diesel", 2.2, "Used", 2021, 32000, 29800],
    ["Volkswagen", "Hatchback", "Petrol", 2.0, "Used", 2015, 110000, 14500],
    ["Toyota", "Pickup", "Diesel", 2.8, "Used", 2020, 55000, 36000],
    ["Ford", "Pickup", "Petrol", 5.0, "New", 2024, 50, 65000],
    ["BMW", "SUV", "Hybrid", 3.0, "Used", 2022, 15000, 72000],
    ["Audi", "Sedan", "Petrol", 2.0, "Used", 2019, 41000, 27500],
    ["Mercedes", "SUV", "Diesel", 3.0, "Used", 2021, 28000, 59000]
]

def get_next_car(index):
    # 1. Try to load from a big external file if it exists
    if os.path.exists(SOURCE_FILE):
        try:
            # Read just the specific row we need (efficient for huge files)
            df = pd.read_csv(SOURCE_FILE, skiprows=index, nrows=1, header=None)
            if not df.empty:
                return df.iloc[0].tolist()
        except Exception:
            pass # Fall back to the starter pack if file fails
            
    # 2. Fall back to the hardcoded real data (Loop over it)
    return REAL_CARS[index % len(REAL_CARS)]

# ---------------- MAIN LOOP ----------------
if __name__ == "__main__":
    print(f"🚗 REAL FEEDER STARTED.")
    print(f"📄 Writing to: {OUTPUT_FILE}")
    print(f"⚡ Speed: {DELAY_SECONDS}s per car")
    print(f"🎯 Target: {TARGET_COUNT} cars")
    
    # Ensure folder exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Create file with headers if missing
    if not os.path.exists(OUTPUT_FILE):
        df = pd.DataFrame(columns=["brand", "cartype", "fueltype", "enginesize", "status", "year", "mileage", "price"])
        df.to_csv(OUTPUT_FILE, index=False)

    count = 0
    try:
        while True:
            # Check target
            if count >= TARGET_COUNT:
                print(f"\n✅ Target of {TARGET_COUNT} cars reached! Stopping.")
                break

            # Get data
            car_data = get_next_car(count)
            
            # Create a DataFrame for the new row
            new_row = pd.DataFrame([car_data], columns=["brand", "cartype", "fueltype", "enginesize", "status", "year", "mileage", "price"])
            
            # Append to CSV
            new_row.to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
            
            # Print every 10th car so the terminal isn't too spammy
            if count % 10 == 0:
                print(f"➕ Added Car #{count+1}: {car_data[5]} {car_data[0]} (${car_data[7]:,})")
            
            count += 1
            time.sleep(DELAY_SECONDS)
            
    except KeyboardInterrupt:
        print("\n🛑 Feeder stopped.")