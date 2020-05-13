import requests

resp = requests.post("http://localhost:5000/predict", files={"file": open('tutorials\_static\img\sample_file.jpeg','rb')})

print(resp.json())