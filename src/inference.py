import json
import mlflow
import mlflow.pytorch

from models.gru import EnsembleGRU

def main():
    top_runs = mlflow.search_runs(
        experiment_ids="387584985157093548",
        order_by=["metrics.val_BinaryMatthewsCorrCoef DESC"],
        max_results=7
    )
    
    log_history = top_runs["tags.mlflow.log-model.history"]
        
    model_uris = []
    for log in log_history:
        model_info = json.loads(log)[0]
        run_id = model_info["run_id"]
        artifact_path = model_info["artifact_path"]
        uri = f"runs:/{run_id}/{artifact_path}"
        model_uris.append(uri)
    
    models = [mlflow.pytorch.load_model(uri) for uri in model_uris]
    
    EnsembleGRU(models, threshold=0.4)