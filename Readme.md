# End-to-End Machine Learning Pipeline with DVC & MLflow

This repository demonstrates a comprehensive, production-grade machine learning workflow using **DVC (Data Version Control)** and **MLflow**. The pipeline is built around the Pima Indians Diabetes Dataset and a Random Forest Classifier, but is easily extensible to other datasets and models. The project emphasizes reproducibility, experiment tracking, and collaborative development—key principles of modern MLOps.

---

## 📈 Project Objectives

- **Reproducibility:** Ensure that every result can be traced and reproduced, regardless of environment or team member.
- **Experimentation:** Enable rapid, organized experimentation with clear tracking of parameters, metrics, and artifacts.
- **Collaboration:** Facilitate teamwork by versioning data, code, and models, and by supporting remote storage.
- **Scalability:** Provide a pipeline structure that can be extended to more complex workflows and larger datasets.

---

## 🏗️ Pipeline Architecture

The pipeline is modular and consists of three main stages, orchestrated by DVC:

### 1. **Preprocessing**
- **Script:** `src/preprocess.py`
- **Input:** `data/raw/data.csv`
- **Output:** `data/processed/data.csv`
- **Description:**  
  - Loads the raw dataset.
  - Cleans and preprocesses data (e.g., renaming columns, handling missing values, feature engineering).
  - Saves the processed dataset for downstream tasks.
- **DVC Stage Example:**
  ```bash
  dvc stage add -n preprocess \
      -d src/preprocess.py -d data/raw/data.csv \
      -o data/processed/data.csv \
      python src/preprocess.py
  ```

### 2. **Training**
- **Script:** `src/train.py`
- **Input:** `data/processed/data.csv`
- **Output:** `models/model.pkl`
- **Description:**  
  - Loads the processed data.
  - Trains a Random Forest Classifier (hyperparameters configurable via `params.yaml`).
  - Saves the trained model.
  - Logs parameters, metrics, and model artifacts to MLflow.
- **DVC Stage Example:**
  ```bash
  dvc stage add -n train \
      -d src/train.py -d data/processed/data.csv \
      -o models/model.pkl \
      python src/train.py
  ```

### 3. **Evaluation**
- **Script:** `src/evaluate.py`
- **Input:** `models/model.pkl`, `data/processed/data.csv`
- **Description:**  
  - Loads the trained model and processed data.
  - Evaluates model performance (e.g., accuracy, confusion matrix).
  - Logs evaluation metrics to MLflow.
- **DVC Stage Example:**
  ```bash
  dvc stage add -n evaluate \
      -d src/evaluate.py -d models/model.pkl -d data/processed/data.csv \
      python src/evaluate.py
  ```

---

## 🗂️ Directory Structure

```
machine_learning_pipeline/
│
├── data/
│   ├── raw/           # Original datasets
│   └── processed/     # Cleaned and feature-engineered datasets
├── models/            # Trained model artifacts
├── src/               # Source code for each pipeline stage
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── dvc.yaml           # DVC pipeline definition
├── dvc.lock           # DVC pipeline lock file
├── params.yaml        # Pipeline and model hyperparameters
├── requirements.txt   # Python dependencies
└── Readme.md
```

---

## ⚙️ Getting Started

### 1. **Clone the Repository**

```bash
git clone <repository-url>
cd machine_learning_pipeline
```

### 2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

### 3. **Download the Dataset**

Place the raw dataset in `data/raw/data.csv`.  
*(You can use the Pima Indians Diabetes Dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database))*.

### 4. **Configure Parameters**

Edit `params.yaml` to adjust preprocessing or model hyperparameters (e.g., number of estimators, max depth).

### 5. **Run the Pipeline**

```bash
dvc repro
```
This will execute all pipeline stages in order, only re-running stages if their dependencies have changed.

### 6. **Track Experiments with MLflow**

Start the MLflow UI to visualize and compare experiment runs:

```bash
mlflow ui
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

### 7. **Version Data and Models Remotely (Optional)**

Configure a DVC remote (e.g., S3, DagsHub) to store large files:

```bash
dvc remote add -d myremote <remote-url>
dvc push
```

---

## 🧪 Experimentation Workflow

- **Change parameters** in `params.yaml` and re-run `dvc repro` to trigger new experiments.
- **Track all runs** in MLflow, including parameters, metrics, and model artifacts.
- **Compare experiments** visually in the MLflow UI to select the best model.
- **Share results** and pipeline state with your team using Git and DVC.

---

## 🔄 Extending the Pipeline

- **Add new models:**  
  Create a new training script (e.g., `train_xgboost.py`), add a new DVC stage, and log results to MLflow.
- **Feature engineering:**  
  Expand `preprocess.py` or add new preprocessing stages.
- **Model deployment:**  
  Integrate deployment scripts or use MLflow’s model serving capabilities.

---

## 📚 References

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 💡 Use Cases

- **Data Science Teams:**  
  Collaborate efficiently with versioned data, code, and experiments.
- **Research & Prototyping:**  
  Rapidly iterate and compare models with full reproducibility.
- **Production ML:**  
  Lay the foundation for robust, production-ready ML workflows.

---

## 📝 License

This project is provided for educational and demonstration purposes.

---
