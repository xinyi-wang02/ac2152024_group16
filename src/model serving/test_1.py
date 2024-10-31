
import requests

resp = requests.post("http://127.0.0.1:5000", files={'file': open("/Users/smallina/Desktop/Stanford/Data/cars_test/00001.jpg", 'rb')})

print(resp.json())