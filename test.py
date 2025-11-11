
import pandas as pd

import requests

df = pd.read_csv("./data/ground-truth-retrieval.csv")
question = df.sample(n=1).iloc[0]['question']

print("question: ", question)

url = "http://localhost:5001/ask"


data = {"question": question}

response = requests.post(url, json=data)

# Check response status and handle errors
print(f"Status Code: {response.status_code}")
print(f"Response Headers: {response.headers}")

if response.status_code == 200:
    try:
        result = response.json()
        print("Response JSON:")
        print(result)
    except requests.exceptions.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Response Text: {response.text}")
else:
    print(f"Error: Received status code {response.status_code}")
    print(f"Response Text: {response.text}")
