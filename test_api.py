import requests

url = "http://127.0.0.1:5000/predict"
# Data matching the heart.csv format
data = {
    "Age": 49,
    "Sex": "F",
    "ChestPainType": "NAP",
    "RestingBP": 160,
    "Cholesterol": 180,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 156,
    "ExerciseAngina": "N",
    "Oldpeak": 1.0,
    "ST_Slope": "Flat"
}

try:
    response = requests.post(url, json=data)
    print("API Response:")
    print(response.json())
except Exception as e:
    print(f"Error connecting to server: {e}")
