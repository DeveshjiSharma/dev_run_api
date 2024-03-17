import requests

url = 'http://localhost:5000/api/chat'  # Updated endpoint
data = {'question': 'Im expericing hairloss and irregular menustral cycles, from last one week, my face has also become noticibliy puffy, no medical conditions, no injuries'}

response = requests.post(url, json=data)

print(response.json())
