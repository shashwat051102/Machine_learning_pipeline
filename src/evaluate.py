import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/shashwatsingh0511/machine_learning_pipeline.mlflow"
os.environ["MLFLOW_TRAcking_USERNAME"] = "shashwatsingh0511"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "99c593c5d3e7fd5f7db3c0e90e70ca536fcd8a7e"



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