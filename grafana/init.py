import os
import json
import requests
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


GRAFANA_URL = "http://localhost:3000"

GRAFANA_USER = os.getenv("GRAFANA_ADMIN_USER")
GRAFANA_PASSWORD = os.getenv("GRAFANA_ADMIN_PASSWORD")

# For Grafana running in Docker, use the service name 'postgres' instead of 'localhost'
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
# If running from host machine, use localhost; if Grafana is in Docker, use 'postgres'
# Since Grafana runs in Docker, we need to use the Docker service name
GRAFANA_POSTGRES_HOST = os.getenv("GRAFANA_POSTGRES_HOST", "postgres")
PG_DB = os.getenv("POSTGRES_DB")
PG_USER = os.getenv("POSTGRES_USER")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD")
PG_PORT = os.getenv("POSTGRES_PORT", "5432")


def get_auth():
    """Returns basic auth tuple for API requests"""
    return (GRAFANA_USER, GRAFANA_PASSWORD)


def create_or_update_datasource():
    auth = get_auth()
    headers = {
        "Content-Type": "application/json",
    }
    datasource_payload = {
        "name": "PostgreSQL",
        "type": "postgres",
        "url": f"{GRAFANA_POSTGRES_HOST}:{PG_PORT}",
        "access": "proxy",
        "user": PG_USER,
        "database": PG_DB,
        "basicAuth": False,
        "isDefault": True,
        "jsonData": {
            "sslmode": "disable",
            "postgresVersion": 1300,
            "database": PG_DB
        },
        "secureJsonData": {"password": PG_PASSWORD},
    }

    print("Datasource payload:")
    print(json.dumps(datasource_payload, indent=2))

    # First, try to get the existing datasource
    response = requests.get(
        f"{GRAFANA_URL}/api/datasources/name/{datasource_payload['name']}",
        auth=auth,
        headers=headers,
    )

    if response.status_code == 200:
        # Datasource exists, let's update it
        existing_datasource = response.json()
        datasource_id = existing_datasource["id"]
        print(f"Updating existing datasource with id: {datasource_id}")
        response = requests.put(
            f"{GRAFANA_URL}/api/datasources/{datasource_id}",
            auth=auth,
            headers=headers,
            json=datasource_payload,
        )
    else:
        # Datasource doesn't exist, create a new one
        print("Creating new datasource")
        response = requests.post(
            f"{GRAFANA_URL}/api/datasources", auth=auth, headers=headers, json=datasource_payload
        )

    print(f"Response status code: {response.status_code}")
    print(f"Response headers: {response.headers}")
    print(f"Response content: {response.text}")

    if response.status_code in [200, 201]:
        print("Datasource created or updated successfully")
        return response.json().get("datasource", {}).get("uid") or response.json().get(
            "uid"
        )
    else:
        print(f"Failed to create or update datasource: {response.text}")
        return None


def create_dashboard(datasource_uid):
    auth = get_auth()
    headers = {
        "Content-Type": "application/json",
    }

    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    dashboard_file = script_dir / "dashboard.json"

    try:
        with open(dashboard_file, "r") as f:
            dashboard_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: {dashboard_file} not found.")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding {dashboard_file}: {str(e)}")
        return

    print("Dashboard JSON loaded successfully.")

    # Update datasource UID in the dashboard JSON
    panels_updated = 0
    for panel in dashboard_json.get("panels", []):
        if isinstance(panel.get("datasource"), dict):
            panel["datasource"]["uid"] = datasource_uid
            panels_updated += 1
        elif isinstance(panel.get("targets"), list):
            for target in panel["targets"]:
                if isinstance(target.get("datasource"), dict):
                    target["datasource"]["uid"] = datasource_uid
                    panels_updated += 1

    print(f"Updated datasource UID for {panels_updated} panels/targets.")

    # Remove keys that shouldn't be included when creating a new dashboard
    dashboard_json.pop("id", None)
    dashboard_json.pop("uid", None)
    dashboard_json.pop("version", None)

    # Prepare the payload
    dashboard_payload = {
        "dashboard": dashboard_json,
        "overwrite": True,
        "message": "Updated by Python script",
    }

    print("Sending dashboard creation request...")

    response = requests.post(
        f"{GRAFANA_URL}/api/dashboards/db", auth=auth, headers=headers, json=dashboard_payload
    )

    print(f"Response status code: {response.status_code}")
    print(f"Response content: {response.text}")

    if response.status_code == 200:
        print("Dashboard created successfully")
        return response.json().get("uid")
    else:
        print(f"Failed to create dashboard: {response.text}")
        return None


def main():
    datasource_uid = create_or_update_datasource()
    if not datasource_uid:
        print("Datasource creation failed")
        return

    create_dashboard(datasource_uid)


if __name__ == "__main__":
    main()