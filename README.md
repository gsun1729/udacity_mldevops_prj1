# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project implements a production-style machine learning pipeline to predict customer churn for a credit card customer base.  
The original notebook-based solution has been refactored into modular Python scripts.

## Files and data description

- **`churn_library.py`**  
  Core library containing all functions required for the churn prediction pipeline, including:
  - Data import
  - EDA generation
  - Feature encoding
  - Feature engineering
  - Model training and evaluation
  - Model persistence

- **`churn_script_logging_and_tests.py`**  
  Script containing basic tests for each major function in `churn_library.py`.  
  Tests verify that:
  - Functions execute without errors
  - Outputs are non-empty
  - Expected artifacts are created on disk  
  Test results and errors are logged to a file.


## Running Files
From the project root directory, execute:

```bash
python churn_library.py
```
Expected behavior:
- Loads the dataset from data/bank_data.csv
- Generates EDA plots in images/eda/
- Performs feature engineering
- Trains Logistic Regression and Random Forest models
- Saves evaluation plots to images/results/
- Saves trained models to models/
- Writes execution logs to logs/churn_library.log

To execute the provided tests:
```bash
python churn_script_logging_and_tests.py
```

Expected behavior:
- Runs basic tests for each function in churn_library.py
- Confirms that required outputs are created
- Logs test execution details to logs/churn_library_tests.log
- Raises errors if any required step fails


