# ML Assignment 2 - Breast Cancer Classification

## a. Problem statement
Build, evaluate, and deploy five classification models on one public dataset. The Streamlit app accepts test CSV data, loads saved models, reports Accuracy, AUC, Precision, Recall, F1, and MCC, and displays a confusion matrix and classification report without retraining.

## b. Dataset description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic)
- **Original source:** UCI Machine Learning Repository; the project retrieves the packaged copy using `sklearn.datasets.load_breast_cancer`.
- **Task:** Binary classification of tumors as malignant or benign.
- **Instances:** 569
- **Features:** 30 numeric predictors computed from digitized fine-needle aspirate images.
- **Target:** `0 = malignant`, `1 = benign`
- **Split:** 80% training and 20% test, stratified, `random_state=42`.

This satisfies the assignment minimum of 500 instances and 12 features.

## c. GitHub Repository Link
Replace after pushing: `https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME`

## d. Models used and comparison table

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.9825 | 0.9954 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |
| Decision Tree          |     0.9035 | 0.9358 |      0.9420 |   0.9028 | 0.9220 | 0.7969 |
| K-Nearest Neighbors    |     0.9737 | 0.9884 |      0.9600 |   1.0000 | 0.9796 | 0.9442 |
| Gaussian Naive Bayes   |     0.9386 | 0.9878 |      0.9452 |   0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine |     0.9825 | 0.9967 |      0.9861 |   0.9861 | 0.9861 | 0.9623 |

### Model-performance observations
- **Logistic Regression**: Accuracy 0.9825, AUC 0.9954, F1 0.9861, and MCC 0.9623. Interpret these values together; AUC measures ranking quality while MCC reflects balanced overall classification.
- **Decision Tree**: Accuracy 0.9035, AUC 0.9358, F1 0.9220, and MCC 0.7969. Interpret these values together; AUC measures ranking quality while MCC reflects balanced overall classification.
- **K-Nearest Neighbors**: Accuracy 0.9737, AUC 0.9884, F1 0.9796, and MCC 0.9442. Interpret these values together; AUC measures ranking quality while MCC reflects balanced overall classification.
- **Gaussian Naive Bayes**: Accuracy 0.9386, AUC 0.9878, F1 0.9517, and MCC 0.8676. Interpret these values together; AUC measures ranking quality while MCC reflects balanced overall classification.
- **Support Vector Machine**: Accuracy 0.9825, AUC 0.9967, F1 0.9861, and MCC 0.9623. Interpret these values together; AUC measures ranking quality while MCC reflects balanced overall classification.

**Overall winner by MCC:** **Logistic Regression**, with MCC **0.9623** on the fixed held-out test split. Re-run `train_models.py` in BITS Virtual Lab and retain its actual console output and screenshot for submission.

## Application features
- Test CSV upload
- Five-model dropdown
- Six required evaluation metrics
- Confusion matrix
- Classification report
- Side-by-side comparison of every saved model
- Prediction CSV download
- Strict column and target validation
- No retraining on uploaded test data

## Repository structure
```text
ml_assignment_2_breast_cancer/
|-- app.py
|-- train_models.py
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

## Run on BITS Virtual Lab
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
Open the displayed local URL, upload `test_data.csv`, evaluate each model, and capture one screenshot showing the BITS Virtual Lab desktop/terminal and the running app.

## Deploy to Streamlit Community Cloud
1. Push every file, including `model/`, to the `main` branch on GitHub.
2. Sign in to Streamlit Community Cloud with GitHub.
3. Click **Create app** or **New app**.
4. Select the repository and `main` branch.
5. Set the entry point to `app.py`.
6. Deploy and test by uploading `test_data.csv`.

## Submission PDF order
1. Clickable GitHub repository link.
2. Clickable live Streamlit app link.
3. One screenshot proving execution on BITS Virtual Lab.
4. Full README content, including the metrics table and observations.

## Reproducibility note
The saved artifacts in `model/` were produced using the pinned package versions in `requirements.txt`. If the environment uses significantly different scikit-learn versions, rerun `python train_models.py` in that environment and commit the regenerated artifacts.
