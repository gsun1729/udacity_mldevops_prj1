'''

Library of functions for a customer churn prediction pipeline:
- Import data
- Perform EDA (save plots)
- Encode categorical variables
- Feature engineering (train/test split)
- Train and evaluate models (save models + plots)

'''


import logging
import os
from typing import List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import GridSearchCV, train_test_split


os.environ['QT_QPA_PLATFORM'] = 'offscreen'

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "churn_library.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

LOGGER = logging.getLogger(__name__)

EDA_DIR = os.path.join("images", "eda")
RESULTS_DIR = os.path.join("images", "results")
MODELS_DIR = "models"


CAT_COLNAMES = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLNAMES = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]
KEEP_COLNAMES = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio',
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn']


def import_data(pth: str) -> pd.DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
        pth: a path to the csv
    output:
        df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    df.drop(columns=['Attrition_Flag'], inplace=True)

    return df


def perform_eda(df: pd.DataFrame, output_dir: str) -> None:
    '''
    perform eda on df and save figures to images folder
    input:
        df: pandas dataframe

    output:
        None
    '''

    def _save(fig_name: str) -> None:
        if output_dir is None:
            out_path = os.path.join(EDA_DIR, fig_name)
        else:
            out_path = os.path.join(output_dir, fig_name)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        LOGGER.info("Saved plot to: %s", out_path)

    assert "Churn" in df.columns, "Churn column not found in DataFrame"
    plt.figure(figsize=(20, 10))
    df["Churn"].hist()
    plt.title("Churn Distribution")
    plt.xlabel("Churn (0=Existing, 1=Attrited)")
    plt.ylabel("Count")
    _save("churn_distribution.png")

    assert "Customer_Age" in df.columns, "Customer_Age column not found in DataFrame"
    plt.figure(figsize=(20, 10))
    df["Customer_Age"].hist()
    plt.title("Customer Age Distribution")
    plt.xlabel("Customer Age")
    plt.ylabel("Frequency")
    _save("customer_age_distribution.png")

    assert "Marital_Status" in df.columns, "Marital_Status column not found in DataFrame"
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.title("Marital Status Distribution")
    plt.xlabel("Marital Status")
    plt.ylabel("Count")
    _save("marital_status_distribution.png")

    assert "Total_Trans_Ct" in df.columns, "Total_Trans_Ct column not found in DataFrame"
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.title("Total Transactions Count Distribution")
    plt.xlabel("Total Transactions Count")
    plt.ylabel("Density")
    _save("total_transactions_count_distribution.png")

    plt.figure(figsize=(20, 10))
    sns.heatmap(df[QUANT_COLNAMES + ["Churn"]].corr(),
                annot=False, cmap='Dark2_r', linewidths=2)
    _save("heatmap.png")


def encoder_helper(
        df: pd.DataFrame,
        category_lst: List[str],
        response: str = "Churn") -> pd.DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
        df: pandas dataframe
        category_lst: list of columns that contain categorical features
        response: string of response name [optional argument that could be used for
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    if response not in df.columns:
        LOGGER.error("Response column '%s' not found in dataframe", response)
        raise ValueError(
            "Response column '%s' not found in dataframe", response)

    out_df = df.copy()

    for col in category_lst:
        LOGGER.info("Encoding column: %s", col)
        if col not in out_df.columns:
            LOGGER.error("Categorical column '%s' not found in dataframe", col)
            raise ValueError(
                "Categorical column '%s' not found in dataframe", col)
        means = out_df.groupby(col)[response].mean()
        out_df[f"{col}_{response}"] = out_df[col].map(means)

    return out_df


def perform_feature_engineering(df: pd.DataFrame,
                                response: str) -> Tuple[pd.DataFrame,
                                                        pd.DataFrame,
                                                        pd.DataFrame,
                                                        pd.DataFrame]:
    '''
    input:
        df: pandas dataframe
        response: string of response name [optional argument that
            could be used for naming variables or index y column]


    output:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    '''

    df = encoder_helper(df, CAT_COLNAMES, response=response)

    Y = df[response]
    X = pd.DataFrame()

    X[KEEP_COLNAMES] = df[KEEP_COLNAMES]
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_report_image(y_train: pd.DataFrame,
                                y_test: pd.DataFrame,
                                y_train_preds_lr: pd.DataFrame,
                                y_train_preds_rf: pd.DataFrame,
                                y_test_preds_lr: pd.DataFrame,
                                y_test_preds_rf: pd.DataFrame,
                                results_dir: str) -> None:
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
        y_train: training response values
        y_test:  test response values
        y_train_preds_lr: training predictions from logistic regression
        y_train_preds_rf: training predictions from random forest
        y_test_preds_lr: test predictions from logistic regression
        y_test_preds_rf: test predictions from random forest

    output:
        None
    '''
    def _save(fig_name: str) -> None:
        if results_dir is None:
            out_path = os.path.join(EDA_DIR, fig_name)
        else:
            out_path = os.path.join(results_dir, fig_name)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        LOGGER.info("Saved plot to: %s", out_path)

    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    _save("rf_results.png")

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    _save("logistic_results.png")


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar")
    write_path = os.path.join(output_pth, "shap_scores.png")
    plt.savefig(write_path)
    plt.close()
    LOGGER.info("Saved plot to: %s", write_path)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    write_path = os.path.join(output_pth, "feature_importance.png")
    plt.tight_layout()
    plt.savefig(write_path)
    plt.close()
    LOGGER.info("Saved plot to: %s", write_path)


def train_models(X_train: pd.DataFrame,
                 x_test: pd.DataFrame,
                 y_train: pd.DataFrame,
                 y_test: pd.DataFrame,
                 results_dir: str,
                 models_dir: str) -> None:
    '''
    train, store model results: images + scores, and store models
    input:
        X_train: X training data
        X_test: X testing data
        y_train: y training data
        y_test: y testing data
    output:
        None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    LOGGER.info("Training models...")

    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    LOGGER.info("Generating predictions (rf)...")
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    LOGGER.info("Generating predictions (lr)...")
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(x_test)

    LOGGER.info('random forest results')
    LOGGER.info('test results')
    LOGGER.info(classification_report(y_test, y_test_preds_rf))
    LOGGER.info('train results')
    LOGGER.info(classification_report(y_train, y_train_preds_rf))

    LOGGER.info('logistic regression results')
    LOGGER.info('test results')
    LOGGER.info(classification_report(y_test, y_test_preds_lr))
    LOGGER.info('train results')
    LOGGER.info(classification_report(y_train, y_train_preds_lr))

    def _save(fig_name: str) -> None:
        if results_dir is None:
            out_path = os.path.join(EDA_DIR, fig_name)
        else:
            out_path = os.path.join(results_dir, fig_name)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        LOGGER.info("Saved plot to: %s", out_path)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                results_dir)

    LOGGER.info("Generating ROC curve comparison plot...")
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_estimator(lrc, x_test, y_test)
    plt.title("ROC Curve – Logistic Regression")
    _save("roc_curve_logistic_regression.png")

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    RocCurveDisplay.from_estimator(
        cv_rfc.best_estimator_, x_test, y_test, ax=ax)
    RocCurveDisplay.from_estimator(lrc, x_test, y_test, ax=ax)
    plt.title("ROC Curves – Random Forest vs Logistic Regression")
    _save("roc_curves_comparison.png")

    LOGGER.info("Saving Models to: %s", models_dir)
    joblib.dump(
        cv_rfc.best_estimator_,
        os.path.join(
            models_dir,
            'rfc_model.pkl'))
    joblib.dump(lrc, os.path.join(models_dir, 'logistic_model.pkl'))

    return cv_rfc.best_estimator_, lrc


def main(df_path: str, eda_path: str, results_path: str, models_path: str)->None:
    """
    Execute the full customer churn modeling pipeline.

    Args:
        df_path (str): File path to the input CSV dataset.
        eda_path (str): Directory path where EDA plots will be saved.
        results_path (str): Directory path where model evaluation artifacts
            (e.g., ROC curves, classification reports) will be saved.
        models_path (str): Directory path where trained model files will be saved.

    Returns:
        None
    """
    LOGGER.info("Starting churn library main function")
    LOGGER.info("Importing data...")
    df = import_data(df_path)
    LOGGER.info("Performing EDA...")
    perform_eda(df, eda_path)
    LOGGER.info("Performing feature engineering...")
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        df, response="Churn")
    LOGGER.info("Training models...")
    rf_model, _ = train_models(
        x_train, x_test, y_train, y_test, results_path, models_path)
    LOGGER.info("Generating feature importance plot...")
    feature_importance_plot(rf_model, x_test, os.path.join(results_path))


if __name__ == "__main__":
    main("data/bank_data.csv",
         "images/eda/",
         "images/results/",
         "models/")
