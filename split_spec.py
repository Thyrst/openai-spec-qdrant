import yaml
import os

with open("openai_openapi.yaml", "r") as file:
    data = yaml.safe_load(file)

os.makedirs("paths", exist_ok=True)

for path, details in data["paths"].items():
    path_data = {path: details}
    file_name = os.path.join("paths", f"{path.strip('/').replace('/', '_')}.yaml")
    with open(file_name, "w") as f:
        yaml.dump(path_data, f)
