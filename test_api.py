import requests
import json
import argparse

def test_model_api(host="http://localhost:5000"):
    """
    Test the model API by sending a sample prediction request
    """
    print(f"Testing model API at {host}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{host}/")
        print(f"Health check: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error checking health: {e}")
    
    # Test info endpoint
    try:
        response = requests.get(f"{host}/info")
        print("\nModel info:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error getting model info: {e}")
    
    # Test prediction endpoint
    sample_input = {
        "no_of_dependents": -0.5,
        "education": 1,
        "self_employed": 0,
        "income_annum": 0.8,
        "loan_amount": 1.2,
        "loan_term": 0.5,
        "cibil_score": 0.9,
        "residential_assets_value": 0.7,
        "commercial_assets_value": 0.4,
        "luxury_assets_value": 0.3,
        "bank_asset_value": 0.6
    }
    
    try:
        response = requests.post(
            f"{host}/predict",
            json=sample_input
        )
        
        print("\nPrediction result:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error making prediction: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the loan approval model API")
    parser.add_argument("--host", default="http://localhost:5000", help="Host URL of the model API")
    args = parser.parse_args()
    
    test_model_api(args.host)
