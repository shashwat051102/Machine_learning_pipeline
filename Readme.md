# Data Pipeline with DVC and MLflow for Machine Learning

This project demonstrates how to build an end-to-end machine learning pipeline using **DVC (Data Version Control)** for data and model versioning, and **MLflow** for experiment tracking. The pipeline focuses on training a Random Forest Classifier on the Pima Indians Diabetes Dataset, with clear stages for data preprocessing, model training, and evaluation.

---

## Key Features

- **Data Version Control (DVC):**
  - Tracks and versions datasets, models, and pipeline stages for full reproducibility.
  - Automatically re-executes pipeline stages if dependencies (data, scripts, parameters) change.
  - Supports remote storage (e.g., DagsHub, S3) for large datasets and models.

- **Experiment Tracking with MLflow:**
  - Logs experiment metrics, parameters, and artifacts.
  - Enables comparison of different model runs and hyperparameters.
  - Facilitates model selection and deployment.

- **Reproducibility:**
  - Ensures that the same data, parameters, and code always produce the same results.

- **Collaboration:**
  - DVC and MLflow enable seamless teamwork, allowing multiple users to track and share changes.

---

## Pipeline Stages

1. **Preprocessing**
    - `src/preprocess.py` reads the raw dataset (`data/raw/data.csv`), performs basic preprocessing (e.g., renaming columns), and outputs the processed data to `data/processed/data.csv`.
    - Ensures consistent data processing across all runs.

2. **Training**
    - `src/train.py` trains a Random Forest Classifier on the preprocessed data.
    - The trained model is saved as `models/model.pkl`.
    - Hyperparameters and model artifacts are logged to MLflow for tracking and comparison.

3. **Evaluation**
    - `src/evaluate.py` loads the trained model and evaluates its performance (e.g., accuracy) on the dataset.
    - Evaluation metrics are logged to MLflow.

---

## Technology Stack

- **Python:** Core programming language for data processing, model training, and evaluation.
- **DVC:** For version control of data, models, and pipeline stages.
- **MLflow:** For logging and tracking experiments, metrics, and model artifacts.
- **Scikit-learn:** For building and training the Random Forest Classifier.

---

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Reproduce the Pipeline

Run the full pipeline using DVC:

```bash
dvc repro
```

### 3. Add or Modify Pipeline Stages

Example commands to add stages:

```bash
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/processed/data.csv \
    python src/preprocess.py

dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/raw/data.csv \
    -o models/model.pkl \
    python src/train.py

dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py
```

### 4. Track Experiments with MLflow

Start the MLflow UI to visualize and compare experiments:

```bash
mlflow ui
```

---

## Use Cases

- **Data Science Teams:** Track datasets, models, and experiments in a reproducible and organized manner.
- **Machine Learning Research:** Iterate quickly over experiments, track performance metrics, and manage data versions effectively.

---

## Project Structure

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

## Goals

- **Reproducibility:** Reliable and consistent results across environments.
- **Experimentation:** Easy tracking and comparison of different experiments.
- **Collaboration:** Smooth teamwork with tracked changes and versioned artifacts.

---

## License

This project is for educational and demonstration purposes.

---
