import requests, os

url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
output_path = "data/instruments/OpenAPIScripMaster.json"

# Create folder if needed
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Download the file
with open(output_path, "wb") as f:
    f.write(requests.get(url).content)

print("✅ Instrument list downloaded successfully.")
