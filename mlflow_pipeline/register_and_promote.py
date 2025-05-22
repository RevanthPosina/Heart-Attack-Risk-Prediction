import os
import sys
import time
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from tabulate import tabulate

EXPERIMENT_NAME = "brfss_heart_attack_risk"
MODEL_NAME = "HeartAttackRiskModel"
MODEL_STATUS_TAG = "best_candidate"
STAGE_DESCRIPTIONS = {
    "Production": "This version is serving live predictions.",
    "Staging": "This version is ready for final validation.",
    "Archived": "This version is archived and not serving."
}

# Set tracking URI
mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001")
os.environ["MLFLOW_TRACKING_URI"] = mlflow_uri
client = MlflowClient(tracking_uri=mlflow_uri)

def color(s, ok=True):
    return f"\033[92m✔︎ {s}\033[0m" if ok else f"\033[91m✗ {s}\033[0m"

def get_latest_best_run():
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"Experiment '{EXPERIMENT_NAME}' not found.")
        sys.exit(1)
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tags.model_status = '{MODEL_STATUS_TAG}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )
    if not runs:
        print(f"No run with tag model_status={MODEL_STATUS_TAG} found.")
        sys.exit(1)
    return runs[0]

def print_versions_table(versions):
    table = []
    for v in sorted(versions, key=lambda x: int(x.version)):
        t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(v.creation_timestamp / 1000))
        table.append([
            v.version, v.current_stage, t, v.run_id, v.status
        ])
    print(tabulate(table, headers=["Version", "Stage", "Created", "Run ID", "Status"]))

def main():
    run = get_latest_best_run()
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    # Register model (if not already registered for this run)
    try:
        result = client.create_registered_model(MODEL_NAME)
        print(color(f"Registered new model: {MODEL_NAME}"))
    except MlflowException:
        pass  # Already exists

    # Check if this run is already registered as a version
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    version_for_run = None
    for v in versions:
        if v.run_id == run_id:
            version_for_run = v
            break

    if version_for_run is None:
        # Register new version
        version_for_run = client.create_model_version(
            name=MODEL_NAME,
            source=client.get_run(run_id).info.artifact_uri + "/model",
            run_id=run_id
        )
        print(color(f"Registered new model version: v{version_for_run.version} for run {run_id}"))
    else:
        print(color(f"Model version for run {run_id} already registered: v{version_for_run.version}"))

    # Transition to Staging
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version_for_run.version,
            stage="Staging",
            archive_existing_versions=False
        )
        print(color(f"Transitioned v{version_for_run.version} to Staging"))
    except MlflowException:
        print(color(f"v{version_for_run.version} already in Staging", ok=False))

    # Promote to Production, archive previous Production if needed
    prod_versions = [v for v in versions if v.current_stage == "Production"]
    if prod_versions:
        for v in prod_versions:
            if v.version != version_for_run.version:
                try:
                    client.transition_model_version_stage(
                        name=MODEL_NAME,
                        version=v.version,
                        stage="Archived"
                    )
                    print(color(f"Archived previous Production v{v.version}"))
                except MlflowException:
                    print(color(f"Failed to archive v{v.version}", ok=False))
    try:
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=version_for_run.version,
            stage="Production"
        )
        print(color(f"Promoted v{version_for_run.version} to Production"))
    except MlflowException:
        print(color(f"v{version_for_run.version} already in Production", ok=False))

    # Update descriptions for each stage (bonus)
    for v in client.search_model_versions(f"name='{MODEL_NAME}'"):
        desc = STAGE_DESCRIPTIONS.get(v.current_stage, "")
        try:
            client.update_model_version(
                name=MODEL_NAME,
                version=v.version,
                description=desc
            )
        except Exception:
            pass

    # Print all versions
    print("\nAll model versions:")
    print_versions_table(client.search_model_versions(f"name='{MODEL_NAME}'"))

if __name__ == "__main__":
    main()