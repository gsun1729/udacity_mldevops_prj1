"""
Tests for churn_library
"""

import os
import logging
import churn_library as cl

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "churn_library_tests.log")

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode="w",
    force=True,
    format="%(name)s - %(levelname)s - %(message)s"
)

LOGGER = logging.getLogger(__name__)


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        LOGGER.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        LOGGER.error("Testing import_data: File not found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        LOGGER.error("Testing import_data: DataFrame empty")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df = cl.import_data("./data/bank_data.csv")
    df = df.sample(n=1000, random_state=0)

    try:
        perform_eda(df, "images/eda/")
        LOGGER.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        LOGGER.error("Testing perform_eda: ERROR")
        raise err

    try:
        assert os.path.exists("images/eda/churn_distribution.png")
    except AssertionError as err:
        LOGGER.error("Testing perform_eda: EDA output missing")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = cl.import_data("./data/bank_data.csv")
    df = df.sample(n=1000, random_state=0)

    try:
        df_encoded = encoder_helper(df, cl.CAT_COLNAMES, response="Churn")
        LOGGER.info("Testing encoder_helper: SUCCESS")
    except Exception as err:
        LOGGER.error("Testing encoder_helper: ERROR")
        raise err

    try:
        for col in cl.CAT_COLNAMES:
            assert f"{col}_Churn" in df_encoded.columns
    except AssertionError as err:
        LOGGER.error("Testing encoder_helper: Encoded columns missing")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df = cl.import_data("./data/bank_data.csv")
    df = df.sample(n=1000, random_state=0)

    try:
        x_train, x_test, y_train, _ = perform_feature_engineering(
            df, response="Churn")
        LOGGER.info("Testing perform_feature_engineering: SUCCESS")
    except Exception as err:
        LOGGER.error("Testing perform_feature_engineering: ERROR")
        raise err

    try:
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] == x_train.shape[0]
    except AssertionError as err:
        LOGGER.error("Testing perform_feature_engineering: Invalid outputs")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    df = cl.import_data("./data/bank_data.csv")
    df = df.sample(n=1000, random_state=0)
    x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
        df, response="Churn")

    os.makedirs("images/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    try:
        train_models(
            x_train,
            x_test,
            y_train,
            y_test,
            "images/results",
            "models"
        )
        LOGGER.info("Testing train_models: SUCCESS")
    except Exception as err:
        LOGGER.error("Testing train_models: ERROR")
        raise err

    try:
        assert os.path.exists("models/rfc_model.pkl")
        assert os.path.exists("models/logistic_model.pkl")
    except AssertionError as err:
        LOGGER.error("Testing train_models: Model artifacts missing")
        raise err


if __name__ == "__main__":
    LOGGER.info("Starting tests...")
    test_import(cl.import_data)
    test_eda(cl.perform_eda)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)
    test_train_models(cl.train_models)

    LOGGER.info("All tests completed successfully.")
