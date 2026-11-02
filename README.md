# ML Assignment 2 - Breast Cancer Classification

## a. Problem statement
Build, evaluate, and deploy five classification models on one public dataset. The app accepts a CSV of test data, loads the saved trained models, and reports the required metrics without retraining. The project includes a Streamlit web application for interactive evaluation and model comparison.

## b. Dataset description
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository, accessed through `sklearn.datasets.load_breast_cancer`
- Task: Binary classification of tumors as malignant or benign
- Instances: 569
- Features: 30 numeric predictors derived from digitized fine-needle aspirate images
- Target: `0 = malignant`, `1 = benign`
- Split: 80% training and 20% testing, stratified with `random_state=42`

This satisfies the assignment requirement of at least 500 instances and 12 features.

## c. GitHub repository link
https://github.com/srinu34/ml_assignment_2_breast_cancer

## d. Live Streamlit application link
https://mlassignment2breastcancer-5fktwpahqiircmacdpzc3d.streamlit.app/

## e. Models used and comparison table
The project trains and saves five models using a fixed reproducible split:
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Support Vector Machine

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |
| Decision Tree          |     0.9035 | 0.9358 |      0.9420 |   0.9028 | 0.9220 | 0.7969 |
| K-Nearest Neighbors    |     0.9737 | 0.9884 |      0.9600 |   1.0000 | 0.9796 | 0.9442 |
| Gaussian Naive Bayes   |     0.9386 | 0.9878 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine |     0.9825 | 0.9967 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |

### Model performance observations
- Logistic Regression achieved an Accuracy of 0.9825, AUC of 0.9954, F1 of 0.9861, and MCC of 0.9623.
- Decision Tree achieved an Accuracy of 0.9035, AUC of 0.9358, F1 of 0.9220, and MCC of 0.7969.
- K-Nearest Neighbors achieved an Accuracy of 0.9737, AUC of 0.9884, F1 of 0.9796, and MCC of 0.9442.
- Gaussian Naive Bayes achieved an Accuracy of 0.9386, AUC of 0.9878, F1 of 0.9517, and MCC of 0.8676.
- Support Vector Machine achieved an Accuracy of 0.9825, AUC of 0.9967, F1 of 0.9861, and MCC of 0.9623.

Overall winner by MCC: Logistic Regression with MCC of 0.9623 on the fixed held-out test split.

## f. Application features
- CSV upload for test data
- Five-model selection dropdown
- Required evaluation metrics: Accuracy, AUC, Precision, Recall, F1, and MCC
- Confusion matrix visualization
- Classification report
- Side-by-side comparison of all saved models
- Prediction download in CSV format
- Strict validation of required feature columns and target column
- No retraining on uploaded test data

## g. Repository structure
```text
ml_assignment_2_breast_cancer/
|-- app.py
|-- api.py
|-- model_utils.py
|-- train_models.py
|-- run_all.py
|-- start_project.sh
|-- Makefile
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- sample_test_data.csv
|-- model_metrics.csv
|-- model_metadata.json
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   `-- svm.joblib
`-- .gitignore
```

## h. How to run locally
### 1) Create and activate virtual environment
```bash
cd "<project-folder>"
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Train the models
```bash
python train_models.py
```

### 4) Run the web app
```bash
streamlit run app.py
```

### 5) Run the FastAPI backend (optional)
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

### 6) Run both app and backend together
```bash
python run_all.py
```

### 7) One-command setup via Makefile
```bash
make install
make train
make run
```

## i. Deploy to Streamlit Community Cloud
1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Click Create app or New app.
4. Select the repository: `srinu34/ml_assignment_2_breast_cancer`.
5. Choose the `main` branch.
6. Set the main file path to `app.py`.
7. Click Deploy.
8. Upload `test_data.csv` or `sample_test_data.csv` to validate the deployment.

Note: Streamlit Community Cloud may temporarily throttle the app when idle. If the app takes a little longer to load, refresh the page after a minute or two.

## j. Submission PDF order
The final submission PDF must follow this order:
1. Clickable GitHub repository link
2. Clickable live Streamlit app link
3. One screenshot proving execution on BITS Virtual Lab
4. Full README content including the metrics table and observations

## k. Reproducibility note
The saved model artifacts in `model/` were produced using the pinned package versions in `requirements.txt`. If the environment differs significantly, rerun `python train_models.py` in that environment and commit the regenerated artifacts.

## l. Final project status
This project has been implemented, trained, and deployed. The app loads the saved trained models and evaluates uploaded CSV data without retraining. The repository contains the complete code and necessary model files for the assignment.

## m. Final links for assignment submission
- GitHub Repository: https://github.com/srinu34/ml_assignment_2_breast_cancer
- Streamlit App: https://mlassignment2breastcancer-5fktwpahqiircmacdpzc3d.streamlit.app/
