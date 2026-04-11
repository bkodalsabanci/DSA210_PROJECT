import pandas as pd
import requests

url = "https://tr.wikipedia.org/wiki/Almanya%27daki_futbol_stadyumlar%C4%B1_listesi"

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

combined.to_csv("germany_stadiums.csv", index=False)