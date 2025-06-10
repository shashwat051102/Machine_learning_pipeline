import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
import mlflow


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/shashwatsingh0511/machine_learning_pipeline.mlflow"
os.environ["MLFLOW_TRAcking_USERNAME"] = "shashwatsingh0511"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "99c593c5d3e7fd5f7db3c0e90e70ca536fcd8a7e"

def hyperparameter_tuning(X_train, y_train, param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search


# Load parameters from param.yaml

params = yaml.safe_load(open('params.yaml'))["train"]

def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    
    X = data.drop(columns=['Outcome'])
    y = data["Outcome"]
    
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
    
    
    # Start the mlflow run
    with mlflow.start_run():
        
        # Split the data set into training anf test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        signature = infer_signature(X_train, y_train)
        
        # Define the hyperprameter grid
        
        param_grid = {
            "n_estimators": [100, 1000],
            "max_depth": [2,4,6,8,10],
            "min_samples_split": [2, 7],
            "min_samples_leaf": [1, 4]
        }
        
        
        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
        
        # Get the best model
        
        best_model = grid_search.best_estimator_
        
        # Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy: {accuracy}")
        
        
        # Log additional metrics
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_metric("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_metric("best_min_samples_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_metric("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
        
        # log the confusion matrix and classification report
        
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        
        mlflow.log_text(cr, "classification_report.txt")
        
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        if tracking_url_type_store != "file":
            # Log the model
            mlflow.sklearn.log_model(best_model, "model",registered_model_name="Best model")
        else:
            mlflow.sklearn.log_model(best_model, "model",signature=signature)
            
        
        # create a directory to save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        filename = model_path
        pickle.dump(best_model, open(filename, 'wb'))
        
        print(f"Model saved to {model_path}")
        
        
    
            
if __name__ == "__main__":
    train(params["data"], params["model"], params["random_state"],params["n_estimators"],params["max_depth"])
    
    