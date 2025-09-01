# server_deploy.py

import requests
import os

def deploy_model_to_server(model_path, server_url_base):
 
    if not os.path.exists(model_path):
        return {"error": f"Model file does not exist: {model_path}"}

    server_url = server_url_base.rstrip("/") + "/deploy-model"

    try:
        with open(model_path, "rb") as f:
            files = {"model_file": f}
            response = requests.post(server_url, files=files)

        if response.status_code == 200:
            return response.json()
        else:
            try:
                return {"error": f"Failed with status code {response.status_code}", "details": response.json()}
            except ValueError:
                return {"error": f"Failed with status code {response.status_code}", "details": response.text}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
