import requests
import json

# URL of your FastAPI server
url = "http://127.0.0.1:8000/"

# Test JSON payload (Iris dataset features)
data = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}

# Convert the data to JSON format
json_data = json.dumps(data)

# Set the headers
headers = {
    "Content-Type": "application/json",
    "accept": "application/json"
}

# Make a POST request to the FastAPI endpoint
response = requests.post(url, data=json_data, headers=headers)

# Print the response
print("Response Code:", response.status_code)
print("Response JSON:", response.json())
