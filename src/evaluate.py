import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = "MLFLOW_TRACKING_URI"
os.environ["MLFLOW_TRAcking_USERNAME"] = "MLFLOW_TRAcking_USERNAME"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "MLFLOW_TRACKING_PASSWORD"



params = yaml.safe_load(open('params.yaml'))["train"]

def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data["Outcome"]
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    # Loading the model from the disk
    
    model = pickle.load(open(model_path, 'rb'))
    
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    
    # log metrics to mlflow
    
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy: {accuracy}")
    
    


if __name__ == "__main__":
    evaluate(params["data"], params["model"])
