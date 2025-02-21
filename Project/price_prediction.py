import pandas as pd
import joblib
import pickle
import numpy as np
from sklearn.preprocessing import  FunctionTransformer
import gzip
import argparse

def load_models():
    tuned_ensemble_model = joblib.load(gzip.open('tuned_ensemble_model.pkl.gz', 'rb'))
    knn_model = joblib.load(open('knn_model.pkl', 'rb'))
    return tuned_ensemble_model, knn_model

def get_default_values():
    """Load default values from df_all_brands.pkl"""
    df = pd.read_pickle("df_all_brands.pkl")
    defaults = {
        "Brand": df["Brand"].mode()[0],  
        "Bike": df["Bike"].mode()[0],    
        "Category": df["Category"].mode()[0],  
        "Power [hp]": df["Power (hp)"].mean(),  
        "Displacement [ccm]": df["Displacement (ccm)"].mean(),  
        "Torque [Nm]": df["Torque (Nm)"].mean(),  
        "Mileage [km]": df["mileage"].mean(),  
        "Age [a]": df["Age"].mean(),  
        "Condition": True,
    }
    return defaults

def get_user_input(defaults):
    """Prompt user for input and use defaults if they press Enter"""
    
    def get_value(prompt, default, data_type):
        user_input = input(f"{prompt} (Press Enter to use default: {default}): ")
        try:
            return (lambda x: x.lower() if isinstance(x, str) else x)(data_type(user_input)) if user_input else default
        except ValueError:
            print(f"Wrong data format! Using default: {default}")
        return default  
    
    brand = get_value("Enter Bike Brand", defaults["Brand"], str)
    bike = get_value("Enter Bike Model", defaults["Bike"], str)
    category = get_value("Enter Bike Category", defaults["Category"], str)
    power = get_value("Enter Power (hp)", defaults["Power [hp]"], float)
    displacement = get_value("Enter Displacement (ccm)", defaults["Displacement [ccm]"], float)
    torque = get_value("Enter Torque (Nm)", defaults["Torque [Nm]"], float)
    mileage = get_value("Enter Mileage (km)", defaults["Mileage [km]"], float)
    age = get_value("Enter Age (years)", defaults["Age [a]"], int)

    return brand, bike, category, power, displacement, torque, mileage, age

def predict_price(Brand=None ,Bike=None , Category=None, Power=None, Displacement=None, Torque=None, Mileage=None, Age=None):
    """Predict price using trained models"""
    target_transformer_log = FunctionTransformer(np.log1p,inverse_func=np.expm1, validate=True)


    # Load models and default values
    tuned_ensemble_model, knn_model = load_models()
    defaults = get_default_values()

    # Fill missing values with defaults
    data = {
        "Brand": Brand if Brand else defaults["Brand"],
        "Bike": Bike if Bike else defaults["Bike"],
        "Category": Category if Category else defaults["Category"],
        "Power [hp]": Power if Power else defaults["Power [hp]"],
        "Displacement [ccm]": Displacement if Displacement else defaults["Displacement [ccm]"],
        "Torque [Nm]": Torque if Torque else defaults["Torque [Nm]"],
        "Mileage [km]": Mileage if Mileage else defaults["Mileage [km]"],
        "Age [a]": Age if Age else defaults["Age [a]"],
        "Condition": Mileage > 100 if Mileage else defaults["Condition"],
    }

    # Convert to DataFrame
    X = pd.DataFrame([data])

    # Predict prices
    y_pred_ens = tuned_ensemble_model.predict(X)
    y_pred_knn = knn_model.predict(X)
    y_pred_ens_inv = target_transformer_log.inverse_transform(y_pred_ens.reshape(-1, 1)).reshape(-1)
    y_pred_knn_inv = target_transformer_log.inverse_transform(y_pred_knn.reshape(-1, 1)).reshape(-1)

    # Convert to float
    ens_price = float(y_pred_ens_inv[0])
    knn_price = float(y_pred_knn_inv[0])

    print(f"Predicted Price (Ensemble Model): ${ens_price:,.2f}")
    print(f"Predicted Price (KNN Model): ${knn_price:,.2f}")

    #return ens_price, knn_price

# Example usage:
#predict_price(Brand="Honda", Bike="CB500", Category="Sport", Power=50, Displacement=471, Torque=43, Mileage=20000, Age=5)
if __name__ == "__main__":
    print("🚀 Welcome to the Bike Price Predictor!")
    defaults = get_default_values()
    user_inputs = get_user_input(defaults)
    predict_price(*user_inputs)