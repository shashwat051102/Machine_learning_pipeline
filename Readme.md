# End-to-End Machine Learning Pipeline with DVC & MLflow

This project showcases a robust, reproducible, and collaborative machine learning workflow using **DVC (Data Version Control)** and **MLflow**. The pipeline demonstrates best practices for data versioning, experiment tracking, and model management, using the Pima Indians Diabetes Dataset and a Random Forest Classifier.

---

## 🚀 Key Highlights

- **Full Pipeline Automation:**  
  Modular pipeline stages for data preprocessing, model training, and evaluation—automatically re-executed when dependencies change.

- **Data & Model Versioning:**  
  DVC tracks every version of your data, models, and pipeline stages, ensuring reproducibility and traceability.

- **Experiment Tracking:**  
  MLflow logs parameters, metrics, and artifacts for every run, making it easy to compare experiments and select the best model.

- **Collaboration Ready:**  
  Seamless teamwork with versioned data, code, and experiments. Supports remote storage (e.g., S3, DagsHub) for large files.

- **Reproducibility by Design:**  
  Every result can be traced back to the exact data, code, and parameters that produced it.

---

## 🛠️ Pipeline Overview

1. **Preprocessing**  
   - Cleans and prepares raw data (`data/raw/data.csv`)  
   - Outputs processed data to `data/processed/data.csv`

2. **Training**  
   - Trains a Random Forest Classifier using scikit-learn  
   - Saves the trained model to `models/model.pkl`  
   - Logs hyperparameters and metrics to MLflow

3. **Evaluation**  
   - Evaluates the trained model on the dataset  
   - Logs evaluation metrics to MLflow

---

## 📦 Project Structure

```
machine_learning_pipeline/
│
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── Readme.md
```

---

## ⚡ Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
dvc repro
```

### 3. Track Experiments with MLflow

```bash
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) in your browser to view and compare experiment runs.

---

## 🧩 Customizing the Pipeline

- **Change Parameters:**  
  Edit `params.yaml` to adjust preprocessing or model hyperparameters.
- **Add New Stages:**  
  Use `dvc stage add` to extend the pipeline with new scripts or models.
- **Remote Storage:**  
  Configure DVC remotes to store large datasets and models in the cloud.

---

## 💡 Use Cases

- **Data Science Teams:**  
  Collaborate efficiently with versioned data, code, and experiments.
- **Research & Prototyping:**  
  Rapidly iterate and compare models with full reproducibility.
- **Production ML:**  
  Lay the foundation for robust, production-ready ML workflows.

---

## 📚 References

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 📝 License

This project is provided for educational and demonstration purposes.

---
