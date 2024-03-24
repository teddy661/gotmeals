import json
from pathlib import Path

import requests

PROTOCOL = "https"
HOST = "edbrown.mids255.com"
PORT = 443
endpoint = f"{PROTOCOL}://{HOST}:{PORT}/predict"

file_to_predict = Path("celery.png")
if not file_to_predict.exists():
    raise FileNotFoundError(f"File {file_to_predict} not found")

headers = {"accept": "application/json"}
files = {"file": open(file_to_predict, "rb")}

response = requests.post(endpoint, files=files, headers=headers)
print(response.status_code)
print(json.dumps(response.json()))
