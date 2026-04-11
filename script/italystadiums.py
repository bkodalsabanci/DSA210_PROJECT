import pandas as pd
import requests

url = "https://en.wikipedia.org/wiki/List_of_Serie_A_stadiums"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

tables = pd.read_html(response.text)

combined = pd.DataFrame()

i = 0
while i < len(tables):
    combined = pd.concat([combined, tables[i]], ignore_index=True)
    i = i + 1

combined.to_csv("italy_stadiums.csv", index=False)