import requests

url = "https://api.apilayer.com/exchangerates_data/latest?symbols=CNY&base=EUR"

payload = {}
headers= {
  "apikey": "lMj4OgUTDQxo9CR2gd2bO7xXOlMSQ7fY"
}

response = requests.request("GET", url, headers=headers, data = payload)

status_code = response.status_code
result = response.text

print(status_code)
print(result)
