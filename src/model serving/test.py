import requests

url = "https://predictions-281462485767.us-east1.run.app"  # Replace with your Cloud Function URL
files = {'file': open('/Users/smallina/Desktop/Stanford/Data/cars_test/00001.jpg', 'rb')}  # Provide a valid image path

resp = requests.post(url, files=files)

print("Status Code:", resp.status_code)
print("Response Text:", resp.text)  # Print raw response for debugging

try:
    print("JSON Response:", resp.json())
except requests.exceptions.JSONDecodeError:
    print("Response is not JSON format.")